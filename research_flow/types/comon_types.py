from typing import TypeAlias

from typing_extensions import TypeVar

ModelType = TypeVar("ModelType")

ModelConfig = TypeVar("ModelConfig")

PatientIdx: TypeAlias = int

Prediction: TypeAlias = list[float]

Real: TypeAlias = list[float]

InType = TypeVar("InType")

OutType = TypeVar("OutType")