from research_flow.types.machine_learning.machine_learning_data_parts_model import (
    MachineLearningDataPartsModel,
)
from research_flow.types.scalars.base_class import Scaler


class MachineLearningDataPartsWithScalersModel(MachineLearningDataPartsModel):
    train_scaler_input_series: Scaler
    train_scaler_output_series: Scaler

    test_scaler_input_series: Scaler
    test_scaler_output_series: Scaler
