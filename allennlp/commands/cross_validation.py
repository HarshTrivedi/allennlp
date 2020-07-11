import copy
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, List, Iterator

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from allennlp.commands.train import TrainModel
from allennlp.common import Lazy, Registrable, util as common_util
from allennlp.data import DataLoader, DatasetReader, Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.training import Trainer, util as training_util
from allennlp.training.metrics import Average
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import lazy_groups_of

logger = logging.getLogger(__name__)


class _MiniPredictManager:
    """
    It's a copy of _PredictManager as in Predict, without all the bells and whistles
    of input file, json etc.
    """

    def __init__(
        self,
        predictor: Predictor,
        output_file: Optional[str],
        batch_size: int
    ) -> None:

        self._predictor = predictor
        self._output_file = open(output_file, "w")
        self._batch_size = batch_size
        self._dataset_reader = predictor._dataset_reader

    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def run(self, instances: List[Instance]) -> None:
        for batch in lazy_groups_of(instances, self._batch_size):
            for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                self._output_file.write(result)
        self._output_file.close()


def instances_get_key(instances: Sequence[Instance], key: str) -> Sequence[Any]:
    return [instance[key] for instance in instances]


@TrainModel.register("cross_validation", constructor="from_partial_objects")
class CrossValidateModel(Registrable):
    def __init__(
        self,
        serialization_dir: str,
        dataset: Dataset,
        dataset_reader: DatasetReader,
        data_loader_builder: Lazy[DataLoader],
        model: Model,
        trainer_builder: Lazy[Trainer],
        dump_test_split_ids: bool = False,
        retrain: bool = False,
        num_splits: int = 5,
        predict: bool = False,
        predictor_name: str = None,
        predict_batch_size: int = 1
    ) -> None:
        self.serialization_dir = serialization_dir
        self.dataset = dataset
        self.data_loader_builder = data_loader_builder
        self.model = model
        self.trainer_builder = trainer_builder
        self.dataset_reader = dataset_reader
        self.dump_test_split_ids = dump_test_split_ids
        self.retrain = retrain
        self.predict = predict
        self.predictor_name = predictor_name
        self.predict_batch_size = predict_batch_size
        self.num_splits = num_splits

    def run(self) -> Dict[str, Any]:
        metrics_by_fold = []

        fold_iterator = KFold(shuffle=True, n_splits=self.num_splits)
        n_splits = fold_iterator.get_n_splits(self.dataset)

        prediction_paths_by_fold = []
        test_indices_by_fold = []

        for fold_index, (non_test_indices, test_indices) in enumerate(
            fold_iterator.split(self.dataset)
        ):

            train_indices = non_test_indices[:len(test_indices)]
            dev_indices = non_test_indices[len(test_indices):]

            logger.info(f"Fold {fold_index}/{n_splits - 1}")

            serialization_dir = os.path.join(self.serialization_dir, f"fold_{fold_index}")
            if common_util.is_master():
                os.makedirs(serialization_dir, exist_ok=True)

            # FIXME: `BucketBatchSampler` needs the dataset to have a vocab, so we workaround it:
            train_dataset = Subset(self.dataset, train_indices)
            train_dataset.vocab = self.dataset.vocab
            train_data_loader = self.data_loader_builder.construct(dataset=train_dataset)

            dev_dataset = Subset(self.dataset, dev_indices)
            dev_dataset.vocab = self.dataset.vocab
            dev_data_loader = self.data_loader_builder.construct(dataset=dev_dataset)

            test_dataset = Subset(self.dataset, test_indices)
            test_dataset.vocab = self.dataset.vocab
            test_data_loader = self.data_loader_builder.construct(dataset=test_dataset)

            model = copy.deepcopy(self.model)
            subtrainer = self.trainer_builder.construct(
                serialization_dir=serialization_dir, data_loader=train_data_loader, 
                validation_data_loader=dev_data_loader, model=model
            )

            fold_metrics = subtrainer.train()

            for metric_key, metric_value in training_util.evaluate(
                model,
                test_data_loader,
                subtrainer.cuda_device,
            ).items():
                if metric_key in fold_metrics:
                    fold_metrics[f"test_{metric_key}"] = metric_value
                else:
                    fold_metrics[metric_key] = metric_value

            if common_util.is_master():
                common_util.dump_metrics(
                    os.path.join(subtrainer._serialization_dir, "metrics.json"),
                    fold_metrics,
                    log=True,
                )

                if self.predict:
                    logger.info("Starting to predict on test instances.")
                    prediction_path = os.path.join(subtrainer._serialization_dir, "predictions.jsonl")
                    predictor = Predictor.by_name(self.predictor_name)(model, self.dataset_reader)
                    predictor_manager = _MiniPredictManager(predictor, prediction_path, self.predict_batch_size)
                    test_instances = [self.dataset[index] for index in test_indices]
                    predictor_manager.run(test_instances)

                if self.dump_test_split_ids:
                    test_instance_ids = [self.dataset.instances[index].fields['metadata']['id']
                                         for index in test_indices]
                    test_instance_ids_path = os.path.join(subtrainer._serialization_dir,
                                                          "test_instance_ids.txt")
                    with open(test_instance_ids_path, "w") as file:
                        file.write("\n".join(test_instance_ids))

            metrics_by_fold.append(fold_metrics)
            prediction_paths_by_fold.append(prediction_path)
            test_indices_by_fold.append(test_indices)

        metrics = {}

        for metric_key, fold_0_metric_value in metrics_by_fold[0].items():
            for fold_index, fold_metrics in enumerate(metrics_by_fold):
                metrics[f"fold{fold_index}_{metric_key}"] = fold_metrics[metric_key]
            if isinstance(fold_0_metric_value, float):
                average = Average()
                for fold_metrics in metrics_by_fold:
                    average(fold_metrics[metric_key])
                metrics[f"average_{metric_key}"] = average.get_metric()

        if self.retrain:
            data_loader = self.data_loader_builder.construct(dataset=self.dataset)

            # We don't need to pass `serialization_dir` and `local_rank` here, because they will
            # have been passed through the trainer by `from_params` already,
            # because they were keyword arguments to construct this class in the first place.
            trainer = self.trainer_builder.construct(model=self.model, data_loader=data_loader)

            trainer.train()

        if self.predict:

            prediction_lines = [None]*len(self.dataset)
            for prediction_path, test_indices in zip(prediction_paths_by_fold, test_indices_by_fold):

                with open(prediction_path) as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]

                for index, line in zip(test_indices, lines):
                    prediction_lines[index] = line

            assert not any(instance is None for instance in prediction_lines)

            merged_prediction_path = os.path.join(self.serialization_dir, "predictions.jsonl")

            with open(merged_prediction_path, "w") as file:
                file.write("\n".join(prediction_lines))

        return metrics

    def finish(self, metrics: Dict[str, Any]) -> None:
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        dataset_reader: DatasetReader,
        data_path: str,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        local_rank: int,  # It's passed transparently directly to the trainer.
        vocabulary: Lazy[Vocabulary] = None,
        dump_test_split_ids: Optional[bool] = None,
        retrain: bool = False,
        num_splits: int = 5,
        predict: bool = False,
        predictor_name: str = None,
        predict_batch_size: int = 1
    ) -> "CrossValidateModel":
        logger.info(f"Reading data from {data_path}")
        dataset = dataset_reader.read(data_path)

        vocabulary_ = vocabulary.construct(instances=dataset) or Vocabulary.from_instances(dataset)
        model_ = model.construct(vocab=vocabulary_)

        if common_util.is_master():
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        dataset.index_with(model_.vocab)

        return cls(
            serialization_dir=serialization_dir,
            dataset=dataset,
            model=model_,
            dataset_reader=dataset_reader,
            data_loader_builder=data_loader,
            trainer_builder=trainer,
            dump_test_split_ids=dump_test_split_ids,
            num_splits=num_splits,
            retrain=retrain,
            predict=predict,
            predictor_name=predictor_name,
            predict_batch_size=predict_batch_size
        )
