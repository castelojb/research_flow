from pydantic import BaseModel


class TrainConfigBaseModel(BaseModel):
    """
    Base model for training configurations.

    This class provides a foundation for defining training configuration models.
    It inherits from Pydantic's BaseModel and can be extended to include
    specific training configuration attributes.

    Attributes:
        None

    Examples:
        To create a custom training configuration model, inherit from this class:

        >>> class CustomTrainConfig(TrainConfigBaseModel):
        >>>    batch_size: int
        >>>    epochs: float
    """

    pass
