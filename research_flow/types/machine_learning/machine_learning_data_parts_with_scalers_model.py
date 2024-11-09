from research_flow.types.machine_learning.machine_learning_data_parts_model import (
    MachineLearningDataPartsModel,
)
from research_flow.types.scalars.base_class import Scaler


class MachineLearningDataPartsWithScalersModel(MachineLearningDataPartsModel):
    """
    Extends the MachineLearningDataPartsModel to include scalers for input and output series.

    This class provides a structure for storing machine learning data parts along with their corresponding scalers.
    The scalers are used to scale the input and output data for training and testing.

    Attributes:
        train_scaler_input_series (Scaler): Scaler for input series used during training.
        train_scaler_output_series (Scaler): Scaler for output series used during training.
        test_scaler_input_series (Scaler): Scaler for input series used during testing.
        test_scaler_output_series (Scaler): Scaler for output series used during testing.
    """

    train_scaler_input_series: Scaler
    train_scaler_output_series: Scaler

    test_scaler_input_series: Scaler
    test_scaler_output_series: Scaler
