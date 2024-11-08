from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from gloe import Transformer
from optuna import Study
from pydantic import BaseModel, ConfigDict

from research_flow.types.comon_types import ModelType, ModelConfig
from research_flow.types.configs.hpo_config_base_model import HPOConfigBaseModel
from research_flow.types.configs.train_config_base_model import TrainConfigBaseModel
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)

DataType = TypeVar("DataType", bound=MachineLearningDataModel)


class MachineLearningAlgorithm(
    BaseModel, Generic[ModelType, ModelConfig, DataType], ABC
):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    alg_factory: Transformer[ModelConfig, ModelType]
    alg_trained: Optional[ModelType] = None
    last_study: Optional[Study] = None

    @abstractmethod
    def fit(
            self,
            data: DataType,
            train_config: TrainConfigBaseModel,
            model_config: ModelConfig,
            val_data: Optional[DataType] = None,
    ) -> "MachineLearningAlgorithm":
        pass

    @abstractmethod
    def predict(self, data: DataType) -> DataType:
        pass

    @abstractmethod
    def tune_model_parameters(
            self,
            data: DataType,
            val_data: DataType,
            hpo_config: HPOConfigBaseModel[ModelConfig],
    ) -> ModelConfig:
        pass

    @abstractmethod
    def save_alg(self, path: str):
        pass

    @abstractmethod
    def load_alg(self, path: str):
        pass
