from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ParameterValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal['number', 'string', 'boolean']


class FunctionValidator(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterValidator]
    returns: ParameterValidator


class PromptValidator(BaseModel):
    prompt: str = Field(..., min_length=1)


class ResultValidator(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parameters: dict[str, float | int | bool | str]
