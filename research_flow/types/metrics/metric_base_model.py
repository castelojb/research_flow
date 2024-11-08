from abc import ABC, abstractmethod
from typing import Generic

from gloe import AsyncTransformer
from pydantic import BaseModel, ConfigDict

from research_flow.machine_learning.base_machine_learning_algorithm import DataType
from research_flow.types.comon_types import PatientIdx
from research_flow.types.metrics.metric_score_model import MetricScoreModel


class MetricBaseModel(BaseModel, Generic[DataType], ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # metric_name: str
    # metric: AsyncTransformer[tuple[PatientIdx, DataType], list[MetricScoreModel]]

    @abstractmethod
    def get_metric_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(
        self,
    ) -> AsyncTransformer[tuple[PatientIdx, DataType], list[MetricScoreModel]]:
        pass
