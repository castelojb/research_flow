from pydantic import BaseModel, field_validator


class MultipleSeriesToSeriesModel(BaseModel):
    input_series: list[list[float]]
    output_series: list[list[float]]

    @field_validator("output_series")
    def check_lengths(cls, v, values):
        """
        The check_lengths function is a validator that ensures the length of the input_series and output_series are equal.

        :param cls: Pass the class of the object to be created
        :param v: Pass the values of the output_series to check_lengths
        :param values: Pass the values of other parameters to this function
        :return: A value
        :doc-author: Trelent
        """
        if "input_series" in values.data and len(v) != len(values.data["input_series"]):
            raise ValueError("input_series and output_series must have the same length")
        return v
