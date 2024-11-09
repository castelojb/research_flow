from abc import ABC, abstractmethod
from typing import Generic, Union, TypeVar

from gloe import Transformer
from pydantic import BaseModel, ConfigDict

from research_flow.types.comon_types import InType, OutType

_U = TypeVar("_U")
_In = TypeVar("_In")
_Out = TypeVar("_Out")


class BaseKernel(BaseModel, Generic[InType, OutType], ABC):

    @property
    @abstractmethod
    def pipeline_graph(self) -> Transformer[InType, OutType]:
        pass

    def transform(self, data: InType) -> OutType:
        return self.pipeline_graph.transform(data)

    def __call__(self, data: InType) -> OutType:
        return self.pipeline_graph(data)

    def __rshift__(self, other: Union["BaseKernel", Transformer]) -> "UnionKernel":

        if isinstance(other, BaseKernel):
            operation2 = other.pipeline_graph

        elif isinstance(other, Transformer):
            operation2 = other
        else:
            raise TypeError(
                f"O objeto `other` deve ser uma instância de BaseKernel ou Transformer. "
                f"Recebido: {type(other).__name__}"
            )

        return UnionKernel(operation1=self.pipeline_graph, operation2=operation2)

    def __call__(self, data: InType) -> OutType:
        return self.pipeline_graph(data)


class UnionKernel(BaseKernel[_In, _Out]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation1: Transformer[_In, _U]

    operation2: Transformer[_U, _Out]

    @property
    def pipeline_graph(self) -> Transformer[_In, _Out]:
        return self.operation1 >> self.operation2
