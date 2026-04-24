from src.utils.validators import ResultValidator, FunctionValidator


class PromptBuilder:
    def __init__(self, decoder, functions_defs):
        self.decoder = decoder
        self._cache: dict[str, ResultValidator] = {}
        self.functions_defs: list[FunctionValidator] = functions_defs

    def build_prompt(self, user_prompt: str) -> str:

        role: str = "role : find the appropriate function to satisfy \
            the user query. You must find function name and parameters " \
            "accordingly to the following functions definition list.\n"

        function_list: list[dict[str, str]] = [f.model_dump() for f in self.functions_defs]
        function_definitions: str = f"function definitions : {function_list}"

        user_query: str = f"user query : {user_prompt}"

        final_prompt: str = role + function_definitions + user_query
        print(final_prompt)
