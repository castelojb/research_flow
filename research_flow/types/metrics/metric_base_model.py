from abc import ABC, abstractmethod
from typing import Generic

from gloe import Transformer
from pydantic import BaseModel, ConfigDict

from research_flow.machine_learning.base_machine_learning_algorithm import DataType
from research_flow.types.comon_types import Score


class MetricBaseModel(BaseModel, Generic[DataType, Score], ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # metric_name: str
    # metric: AsyncTransformer[tuple[PatientIdx, DataType], list[MetricScoreModel]]

    @abstractmethod
    def get_metric_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(
        self,
    ) -> Transformer[DataType, Score]:
        pass
