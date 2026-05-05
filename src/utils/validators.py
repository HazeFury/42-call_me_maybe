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
    Verifies the type of parameters for each generated result and casts them
    into the correct Python native types based on the function definitions.
    """
    # OPTIMIZATION: Map function names to their definitions for O(1) lookups.
    # This avoids iterating over the function_defs list for every single prompt
    functions_map = {func.name: func for func in function_defs}

    formatted_results: list[dict[str, Any]] = []

    for item in content:
        # Create a shallow copy to avoid mutating the original input data
        formatted_item = item.copy()
        func_name = formatted_item.get("name")
        raw_parameters = formatted_item.get("parameters", {})

        # If the generated function name exists in our definitions catalog
        if func_name in functions_map:
            expected_params_def = functions_map[func_name].parameters
            casted_parameters = {}

            for param_name, raw_value in raw_parameters.items():
                # Check if the generated parameter is actually expected
                if param_name in expected_params_def:
                    expected_type = expected_params_def[param_name].type

                    try:
                        # Perform the strict casting based on defined rules
                        if expected_type in ["number", "float"]:
                            casted_parameters[param_name] = float(raw_value)

                        elif expected_type == "integer":
                            casted_parameters[param_name] = int(raw_value)

                        elif expected_type == "boolean":
                            # Handle potential string generations like "true"
                            if isinstance(raw_value, str):
                                casted_parameters[param_name] = \
                                    raw_value.strip().lower() in [
                                        "true", "1", "yes"
                                        ]
                            else:
                                casted_parameters[param_name] = bool(raw_value)

                        else:
                            # Default fallback for 'string' or any unknown type
                            casted_parameters[param_name] = str(raw_value)

                    except (ValueError, TypeError):
                        # Safety net: if casting fails (ex: float("abc")),
                        # we keep the raw value to avoid crashing the whole
                        # pipeline.
                        casted_parameters[param_name] = raw_value
                else:
                    # If the LLM hallucinated an unknown param, we keep it raw
                    casted_parameters[param_name] = raw_value

            # Replace the raw parameters with our safely casted parameters
            formatted_item["parameters"] = casted_parameters

        formatted_results.append(formatted_item)

    return formatted_results
