from pydantic import BaseModel

from research_flow.types.machine_learning.machine_learning_data_model import MachineLearningDataModel


class MachineLearningDataPartsModel(BaseModel):
    train: MachineLearningDataModel

    validation: MachineLearningDataModel

    test: MachineLearningDataModel
