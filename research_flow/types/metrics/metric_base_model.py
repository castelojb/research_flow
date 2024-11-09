from abc import ABC, abstractmethod
from typing import Generic

from gloe import Transformer
from pydantic import BaseModel, ConfigDict

from research_flow.machine_learning.base_machine_learning_algorithm import DataType
from research_flow.types.metrics.metric_score_model import MetricScoreModel


class MetricBaseModel(BaseModel, Generic[DataType], ABC):
    """
    Base class for representing a metric in a machine learning context.

    This class provides a blueprint for creating specific metric models that can be used to calculate metric scores for various datasets.

    The `MetricBaseModel` class is generic over `DataType`, allowing it to work with different types of datasets.

    The `model_config` attribute is a configuration dictionary that allows arbitrary types.

    The `MetricBaseModel` class has two abstract methods:

    - `get_metric_name`: Returns the name of the metric as a string.
    - `get_metric`: Returns a transformer that calculates the metric score for a given dataset.

    The transformer returned by `get_metric` takes in a dataset of type `DataType` and returns a `MetricScoreModel` containing the metric score.

    Subclasses of `MetricBaseModel` must implement the abstract methods `get_metric_name` and `get_metric` to provide the specific metric calculation logic.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name of the metric.
        """
        pass

    @abstractmethod
    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScoreModel]:
        """
        Returns a transformer that calculates the metric score for a given dataset.

        The returned transformer takes in a dataset of type `DataType` and returns a
        `MetricScoreModel` containing the metric score.

        Returns:
            Transformer[DataType, MetricScoreModel]: The transformer that calculates the metric score.
        """
        pass
