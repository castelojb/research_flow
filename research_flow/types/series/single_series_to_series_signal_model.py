import pydantic

from research_flow.types.series.single_series_to_series_model import SingleSeriesToSeriesModel


class SingleSeriesToSeriesSignalModel(SingleSeriesToSeriesModel):
    input_series_frequency: pydantic.PositiveInt

    output_series_frequency: pydantic.PositiveInt
