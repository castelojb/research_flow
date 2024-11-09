from research_flow.types.series.multiple_series_to_series_model import (
    MultipleSeriesToSeriesModel,
)


class MachineLearningDataModel(MultipleSeriesToSeriesModel):
    """
    A data model for machine learning tasks, extending the MultipleSeriesToSeriesModel.

    Attributes:
        prediction (list[list[float]] | None): The predicted output, represented as a list of lists of floats. Defaults to None.
    """

    prediction: list[list[float]] | None = None
