from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.prompt_builder import PromptBuilder
from src.utils.validators import (
        ResultValidator,
        PromptValidator,
        FunctionValidator
)
from src.core.constrained_decoder import ConstrainedDecoder
from src.core.vocab_manager import VocabManager


class GenerationOrchestrator:
    def __init__(self, llm: Small_LLM_Model, prompter: PromptBuilder):
        self.llm: Small_LLM_Model = llm
        self.prompter: PromptBuilder = prompter
        self._cache: dict[str, ResultValidator] = {}

    def run_generation(
            self,
            prompts: list[PromptValidator],
            functions: list[FunctionValidator]
            ) -> None:
        """Iterates over all prompts and runs the basic LLM generation loop."""
        vocab_path = self.llm.get_path_to_vocab_file()
        vocab_manager = VocabManager(vocab_path)

        decoder = ConstrainedDecoder(
            vocab_manager=vocab_manager,
            functions_defs=functions
            )

        for prompt in prompts:
            current_prompt = self.prompter.build_prompt(prompt)
            print(f"--- Processing query: {prompt.prompt} ---")

            input_tensor = self.llm.encode(current_prompt)
            input_ids = input_tensor[0].tolist()

            print("Output: ", end="", flush=True)
            decoder.reset_state()

            while True:
                logits = self.llm.get_logits_from_input_ids(input_ids)

                logits = decoder.filter_logits(logits)

                next_token_id = logits.index(max(logits))
                input_ids.append(next_token_id)

                new_word = self.llm.decode([next_token_id])
                decoder.update_state(new_word)

                print(new_word, end="", flush=True)
                if decoder.current_state == "DONE":
                    break
            print("\n" + "="*50 + "\n")
