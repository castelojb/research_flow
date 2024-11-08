from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from pydantic import BaseModel


class Scaler(BaseModel, ABC):
    FLATTEN_LIST: ClassVar[str] = "list[float]"
    MATRICE: ClassVar[str] = "list[list[float]]"

    @abstractmethod
    async def fit(self, x: list[float] | list[list[float]]):
        pass

    @abstractmethod
    async def transform(
        self, x: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        pass

    @abstractmethod
    async def inverse_transform(
        self, x: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        pass

    @abstractmethod
    async def copy_empty_like(self) -> "Scaler":
        pass

    @staticmethod
    async def resolve_type(
        x: list[float] | list[list[float]],
    ) -> Optional[str]:
        if isinstance(x, list):
            if all(isinstance(i, float) for i in x):
                return Scaler.FLATTEN_LIST
            elif all(
                isinstance(i, list) and all(isinstance(j, float) for j in i) for i in x
            ):
                return Scaler.MATRICE
        return None
