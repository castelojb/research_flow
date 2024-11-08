import pydantic

from research_flow.types.series.multiple_series_to_series_model import MultipleSeriesToSeriesModel


class MultipleSeriesToSeriesSignalModel(MultipleSeriesToSeriesModel):
    input_series_frequency: pydantic.PositiveInt

    output_series_frequency: pydantic.PositiveInt
