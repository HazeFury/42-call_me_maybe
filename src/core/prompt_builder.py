import json
from llm_sdk import Small_LLM_Model  # type: ignore
from src.utils.validators import FunctionValidator, PromptValidator


class PromptBuilder:
    """Builds the system prompt injected into the LLM context."""

    def __init__(self,
                 llm: Small_LLM_Model,
                 functions_defs: list[FunctionValidator]):
        functions_dicts = [f.model_dump() for f in functions_defs]
        self.functions_definitions_json: str = json.dumps(
            functions_dicts, indent=2
            )

    def build_prompt(self) -> str:
        """Formats the final string to send to the LLM."""

        role: str = (
            "System: Function calling."
        )

        function_definitions: str = "function definitions:\n" \
                                    f"{self.functions_definitions_json}\n\n"

        final_prompt: str = role + function_definitions
        return final_prompt

    def prepare_user_query(self, user_prompt: PromptValidator) -> str:
        """
        Add the current user query to the global prompt to avoid
        recalculating the whole prompt.
        """
        user_query: str = f"user query: {user_prompt.prompt}\n"

        start: str = 'start : ```JSON\n{ "name": "'

        final_prompt: str = user_query + start
        return final_prompt
