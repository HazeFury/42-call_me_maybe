from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.prompt_builder import PromptBuilder
from src.utils.validators import (
        ResultValidator,
        PromptValidator,
)


class GenerationOrchestrator:
    def __init__(self, llm: Small_LLM_Model, prompter: PromptBuilder):
        self.llm: Small_LLM_Model = llm
        self.prompter: PromptBuilder = prompter
        self._cache: dict[str, ResultValidator] = {}

    def run_generation(self, prompts: list[PromptValidator]) -> None:
        """Iterates over all prompts and runs the basic LLM generation loop."""

        for prompt in prompts:
            current_prompt = self.prompter.build_prompt(prompt)
            print(f"--- Processing query: {prompt.prompt} ---")

            # 1. Encode text to tokens
            input_tensor = self.llm.encode(current_prompt)
            input_ids = input_tensor[0].tolist()
            print(f"debut : {input_ids}\n\n")
            print("Output: ", end="", flush=True)

            # 2. Generate tokens one by one (max 60 for testing)
            for _ in range(1):
                logits = self.llm.get_logits_from_input_ids(input_ids)

                # Greedy decoding: taking the highest probability token
                next_token_id = logits.index(max(logits))
                input_ids.append(next_token_id)
                print(f"apres : {input_ids}\n\n")

                # Decode and print the new token directly
                new_word = self.llm.decode([next_token_id])
                print(new_word, end="", flush=True)

            print("\n" + "="*50 + "\n")
