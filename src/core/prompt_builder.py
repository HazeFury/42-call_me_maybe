import json
from src.utils.validators import FunctionValidator, PromptValidator


class PromptBuilder:
    """Builds the system prompt injected into the LLM context."""

    def __init__(self, functions_defs: list[FunctionValidator]):
        functions_dicts = [f.model_dump() for f in functions_defs]
        self.functions_definitions_json: str = json.dumps(
            functions_dicts, indent=2
            )

    def build_prompt(self, user_prompt: PromptValidator) -> str:
        """Formats the final string to send to the LLM."""

        role: str = (
            "role: find the appropriate function to satisfy the user query. "
            "You must find function name and parameters accordingly to the "
            "following functions definition list. "
            "You must reply ONLY with a valid JSON object containing the "
            "keys 'name' and 'parameters'. Do not add any text before "
            "or after.\n\n"
        )

        function_definitions: str = "function definitions:\n" \
                                    f"{self.functions_definitions_json}\n\n"

        user_query: str = f"user query: {user_prompt.prompt}\n"

        final_prompt: str = role + function_definitions + user_query
        return final_prompt
