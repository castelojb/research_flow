from typing import Generic

from pydantic import BaseModel

from research_flow.types.comon_types import ModelConfig


class HPOConfigBaseModel(BaseModel, Generic[ModelConfig]):
    """
    A base model for Hyperparameter Optimization (HPO) configurations.

    This model serves as a foundation for defining HPO configurations and can be
    extended to include specific fields and validation logic.

    Args:
        ModelConfig: The type of model configuration being optimized.

    Note:
        This is an abstract base model and should not be instantiated directly.
    """

    pass
