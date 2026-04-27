from src.utils.validators import ResultValidator, FunctionValidator


class PromptBuilder:
    def __init__(self, functions_defs, decoder: bool = False):
        self.decoder = decoder
        self._cache: dict[str, ResultValidator] = {}
        self.functions_defs: list[FunctionValidator] = functions_defs

    def build_prompt(self, user_prompt: str) -> str:

        role: str = "role : find the appropriate function to satisfy " \
            "the user query. You must find function name and parameters " \
            "accordingly to the following functions definition list. "\
            "You must reply ONLY with a valid JSON object containing the " \
            "keys 'name' and 'parameters'. Do not add any text before " \
            "or after.\n"

        function_list: list[dict[str, str]] = [
            f.model_dump_json() for f in self.functions_defs
            ]
        function_definitions: str = f"function definitions : {function_list}\n"

        user_query: str = f"user query : {user_prompt}\n"

        final_prompt: str = role + function_definitions + user_query
        print(final_prompt)
        return final_prompt
