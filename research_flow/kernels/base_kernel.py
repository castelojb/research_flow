from abc import ABC, abstractmethod
from typing import Generic, Union, TypeVar

from gloe import Transformer
from pydantic import BaseModel

from research_flow.types.comon_types import InType, OutType

_U = TypeVar("_U")


class BaseKernel(BaseModel, Generic[InType, OutType], ABC):

    @property
    @abstractmethod
    def pipeline_graph(self) -> Transformer[InType, OutType]:
        pass

    def transform(self, data: InType) -> OutType:
        return self.pipeline_graph.transform(data)

    def __rshift__(self, other: Union["BaseKernel", Transformer]):

        if isinstance(other, BaseKernel):
            return self.pipeline_graph.__rshift__(other.pipeline_graph)

        elif isinstance(other, Transformer):
            return self.pipeline_graph.__rshift__(other)

        else:
            raise TypeError(
                f"O objeto `other` deve ser uma instância de BaseKernel ou Transformer. "
                f"Recebido: {type(other).__name__}"
            )

    def __call__(self, data: InType) -> OutType:
        return self.pipeline_graph(data)
