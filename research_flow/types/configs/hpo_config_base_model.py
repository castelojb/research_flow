from typing import Generic

from pydantic import BaseModel

from research_flow.types.comon_types import ModelConfig


class HPOConfigBaseModel(BaseModel, Generic[ModelConfig]):
    pass
