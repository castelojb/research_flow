from pydantic import BaseModel


class SingleSeriesToSeriesModel(BaseModel):

    input_series: list[float]
    output_series: list[float]
