from pydantic import BaseModel, PositiveInt


class SingleSeriesPeaksModel(BaseModel):
    input_series_peaks: list[PositiveInt]
    output_series_peaks: list[PositiveInt]
