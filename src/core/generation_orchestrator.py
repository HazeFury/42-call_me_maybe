from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.prompt_builder import PromptBuilder
from src.utils.validators import (
        ResultValidator,
        PromptValidator,
        FunctionValidator
)
from src.core.constrained_decoder import ConstrainedDecoder
from src.core.vocab_manager import VocabManager
from src.utils.file_handler import format_final_result
from typing import Any


class GenerationOrchestrator:
    def __init__(self, llm: Small_LLM_Model, prompter: PromptBuilder):
        self.llm: Small_LLM_Model = llm
        self.prompter: PromptBuilder = prompter
        self._cache: dict[str, ResultValidator] = {}
        self.input_ids: Any = []

    def add_tokens_to_context_if_possible(
            self,
            decoder: ConstrainedDecoder
            ) -> bool:
        """
        Add tokens to context if the state permit it in order to gain time.
        """
        if decoder.current_state == "PARAM_KEY":
            current_param = decoder.params_queue.pop(0)

            param_text = f' "{current_param[0]}": '
            param_tensor = self.llm.encode(param_text)
            param_ids = param_tensor[0].tolist()
            [self.input_ids.append(x) for x in param_ids]
            decoder.update_state(param_text)
            return True

        elif decoder.current_state == "PARAMS_KEY":
            parameter_text: str = ', "parameters": {'

            parameter_tensor = self.llm.encode(parameter_text)
            parameter_ids = parameter_tensor[0].tolist()
            [self.input_ids.append(x) for x in parameter_ids]
            decoder.update_state(parameter_text)
            return True

        elif decoder.current_state == "CLOSING_BRACE":
            decoder.update_state(" }")
            return True

        return False

    def run_generation(
            self,
            prompts: list[PromptValidator],
            functions: list[FunctionValidator]
            ) -> list[dict[str, str | int]]:
        """Iterates over all prompts and runs the basic LLM generation loop."""
        vocab_path = self.llm.get_path_to_vocab_file()
        vocab_manager = VocabManager(vocab_path)

        decoder = ConstrainedDecoder(
            vocab_manager=vocab_manager,
            functions_defs=functions
            )
        result: list[dict[str, str | int]] = []

        for prompt in prompts:
            current_prompt = self.prompter.build_prompt(prompt)
            print(f"--- Processing query: {prompt.prompt} ---")

            input_tensor = self.llm.encode(current_prompt)
            self.input_ids = input_tensor[0].tolist()

            # print("Output: ", end="", flush=True)
            decoder.reset_state()

            i = 0
            while decoder.current_state != "DONE":
                print(f"#{i} : '{decoder.generated_text}'")
                if self.add_tokens_to_context_if_possible(decoder):
                    continue
                logits = self.llm.get_logits_from_input_ids(self.input_ids)

                logits = decoder.filter_logits(logits)

                next_token_id = logits.index(max(logits))
                self.input_ids.append(next_token_id)

                new_word = self.llm.decode([next_token_id])
                decoder.update_state(new_word)
                i += 1
                # print(new_word, end="", flush=True)
            print("\n" + "="*50 + "\n")
            print(f"#{i} : '{decoder.generated_text}'")
            tmp = format_final_result(prompt, decoder.generated_text)
            result.append(tmp)

        return result
