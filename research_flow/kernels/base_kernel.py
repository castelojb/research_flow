from abc import ABC, abstractmethod, abstractproperty
from typing import Generic

from gloe import Transformer, transformer
from pydantic import BaseModel

from research_flow.types.comon_types import InType, OutType


class BaseKernel(BaseModel, Generic[InType, OutType], Transformer[InType, OutType], ABC):

    @property
    @abstractmethod
    def pipeline_graph(self) -> Transformer[InType, OutType]:
        pass

    def transform(self, data: InType) -> OutType:
        return self.pipeline_graph(data)