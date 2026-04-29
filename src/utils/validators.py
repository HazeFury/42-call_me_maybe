from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any


class ParameterValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal['number', 'string', 'boolean', "integer", "float"]


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


def format_and_cast_results(
        content: list[dict[str, Any]],
        function_defs: list[FunctionValidator]
        ) -> list[dict[str, Any]]:
    """
    Verifies the type of parameters for each result and casts them if
    necessary.
    """
    pass
