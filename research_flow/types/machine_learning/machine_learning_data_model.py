from research_flow.types.series.multiple_series_to_series_model import MultipleSeriesToSeriesModel


class MachineLearningDataModel(MultipleSeriesToSeriesModel):
    prediction: list[list[float]] | None = None
