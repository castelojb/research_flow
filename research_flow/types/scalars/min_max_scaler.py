from itertools import chain

import numpy as np

from research_flow.types.scalars.base_class import Scaler


class MinMaxScalerModel(Scaler):
    min_feature: int
    max_feature: int

    min_value: float | None = None
    max_value: float | None = None

    def fit(self, x: list[float] | list[list[float]]):
        entry_type = self.resolve_type(x)

        if entry_type == self.FLATTEN_LIST:
            self.min_value = min(x)  # type: ignore
            self.max_value = max(x)  # type: ignore

        if entry_type == self.MATRICE:
            flatten = list(chain(*x))
            self.min_value = min(flatten)
            self.max_value = max(flatten)

    def transform(
        self, x: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        x_array = np.array(x)

        x_std = (x_array - self.min_value) / (self.max_value - self.min_value)
        x_scaled = x_std * (self.max_feature - self.min_feature) + self.min_feature

        return x_scaled.tolist()

    def inverse_transform(
        self, x_scaled: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        x_scaled_array = np.array(x_scaled)

        x_std = (x_scaled_array - self.min_feature) / (
            self.max_feature - self.min_feature
        )
        x = x_std * (self.max_value - self.min_value) + self.min_value

        return x.tolist()

    def copy_empty_like(self) -> "MinMaxScalerModel":
        return MinMaxScalerModel(
            min_feature=self.min_feature, max_feature=self.max_feature
        )
