from pydantic import BaseModel

from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)


class MachineLearningDataPartsModel(BaseModel):
    """
    Represents the different parts of a machine learning dataset.

    Attributes:
        train (MachineLearningDataModel): The training set.
        validation (MachineLearningDataModel): The validation set.
        test (MachineLearningDataModel): The test set.
    """

    train: MachineLearningDataModel

    validation: MachineLearningDataModel

    test: MachineLearningDataModel
