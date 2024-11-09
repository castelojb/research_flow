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
        """
        Trains a machine learning model.

        Args:
        - data: The data to train the model on.
        - train_config: The configuration for the training process.
        - model_config: The configuration of the model to be trained.
        - val_data: Optional data to validate the model against.

        Returns:
        - self
        """
        pass

    @abstractmethod
    def predict(self, data: DataType) -> DataType:
        """
        Makes a prediction using the trained model.

        Args:
        - data: The data to make a prediction on.

        Returns:
        - A prediction in the form of a DataType.
        """
        pass

    @abstractmethod
    def tune_model_parameters(
        self,
        data: DataType,
        val_data: DataType,
        hpo_config: HPOConfigBaseModel[ModelConfig],
    ) -> ModelConfig:
        """
        Tunes the parameters of the machine learning model using hyperparameter optimization.

        Args:
        - data: The data to tune the model on.
        - val_data: The validation data to tune the model against.
        - hpo_config: The configuration for the hyperparameter optimization process.

        Returns:
        - The best model configuration found during hyperparameter optimization.
        """
        pass

    @abstractmethod
    def save_alg(self, path: str):
        """
        Saves the machine learning algorithm to the specified file path.

        Args:
        - path: The file path where the algorithm will be saved.
        """
        pass

    @abstractmethod
    def load_alg(self, path: str):
        """
        Loads a saved machine learning algorithm from the specified file path.

        Args:
        - path: The file path from which the algorithm will be loaded.

        Returns:
        - The loaded machine learning algorithm.
        """
        pass
