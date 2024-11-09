from abc import ABC, abstractmethod
from typing import Generic, Union, TypeVar

from gloe import Transformer
from pydantic import BaseModel, ConfigDict

from research_flow.types.comon_types import InType, OutType

_U = TypeVar("_U")
_In = TypeVar("_In")
_Out = TypeVar("_Out")


class BaseKernel(BaseModel, Generic[InType, OutType], ABC):
    """
    Abstract base class for kernels that transform input data.

    A kernel is a transformation pipeline that takes input data of type `InType` and produces output data of type `OutType`.
    This class provides a framework for defining and chaining kernels.

    Subclasses should implement the `pipeline_graph` property to define their own transformation logic.

    Attributes:
        pipeline_graph: A valid `Transformer` object from the `gloe` package that defines the transformation pipeline.

    Methods:
        transform(data: InType) -> OutType: Applies the pipeline graph to the input data.
        __call__(data: InType) -> OutType: Applies the pipeline graph to the input data (equivalent to `transform`).
        __rshift__(other: Union["BaseKernel", Transformer]) -> "UnionKernel": Chains multiple kernels together.

    Notes:
        This class is abstract and cannot be instantiated directly.
    """

    @property
    @abstractmethod
    def pipeline_graph(self) -> Transformer[InType, OutType]:
        """
        Should return a valid `Transformer` object from `gloe` package.
        This `Transformer` object is used for transforming input data.
        """
        pass

    def transform(self, data: InType) -> OutType:
        """
        Apply the pipeline graph to `data`.

        This method is equivalent to calling `self.pipeline_graph(data)`.
        """
        return self.pipeline_graph.transform(data)

    def __call__(self, data: InType) -> OutType:
        """
        Apply the pipeline graph to `data`.

        This method is equivalent to calling `self.pipeline_graph(data)`.

        Parameters
        ----------
        data: InType
            Input data to be transformed.

        Returns
        -------
        OutType
            Transformed output data.
        """
        return self.pipeline_graph(data)

    def __rshift__(self, other: Union["BaseKernel", Transformer]) -> "UnionKernel":
        """
        Overload the `__rshift__` operator.

        This method allows you to chain multiple `BaseKernel` instances together.
        It takes another `BaseKernel` or `Transformer` instance and returns a new
        `UnionKernel` instance that represents the composition of the two.

        Parameters
        ----------
        other: Union["BaseKernel", Transformer]
            The other `BaseKernel` or `Transformer` instance to be composed.

        Returns
        -------
        UnionKernel
            A new `UnionKernel` instance representing the composition of `self` and `other`.

        Notes
        -----
        This method is used to implement the `>>` operator.
        """
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


class UnionKernel(BaseKernel[_In, _Out]):
    """
    A kernel that combines two transformer operations into a single pipeline.

    Attributes:
        operation1 (Transformer[_In, _U]): The first transformer operation in the pipeline.
        operation2 (Transformer[_U, _Out]): The second transformer operation in the pipeline.
        model_config (ConfigDict): A configuration dictionary that allows arbitrary types.

    Properties:
        pipeline_graph (Transformer[_In, _Out]): The composed transformer pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation1: Transformer[_In, _U]

    operation2: Transformer[_U, _Out]

    @property
    def pipeline_graph(self) -> Transformer[_In, _Out]:
        return self.operation1 >> self.operation2
