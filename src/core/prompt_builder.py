class PromptBuilder:
    def __init__(self, functions_defs: list[dict[str, str]]):
        self.functions_defs: list[dict[str, str]] = functions_defs

    def build_prompt(self, user_prompt: str) -> str:

        role: str = "role : find the appropriate function to satisfy " \
            "the user query. You must find function name and parameters " \
            "accordingly to the following functions definition list. "\
            "You must reply ONLY with a valid JSON object containing the " \
            "keys 'name' and 'parameters'. Do not add any text before " \
            "or after.\n"

        function_definitions: str = "function definitions :" \
                                    f" {self.functions_defs}\n"

        prompt = user_prompt.replace('{"prompt":"', "").replace('"}', "")
        user_query: str = f"user query : {prompt}\n"

        final_prompt: str = role + function_definitions + user_query
        return final_prompt
