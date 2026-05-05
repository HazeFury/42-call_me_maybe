import numpy as np
import time
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
            if decoder.params_queue:
                decoder.current_param = decoder.params_queue.pop(0)

            param_text = f' "{decoder.current_param[0]}": '
            param_tensor = self.llm.encode(param_text)
            param_ids = param_tensor[0].tolist()

            self.input_ids.extend(param_ids)
            decoder.update_state(param_text)
            return True

        elif decoder.current_state == "PARAMS_KEY":
            parameter_text: str = ', "parameters": {'

            parameter_tensor = self.llm.encode(parameter_text)
            parameter_ids = parameter_tensor[0].tolist()
            self.input_ids.extend(parameter_ids)
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

        global_prompt = self.prompter.build_prompt()

        global_input_tensor = self.llm.encode(global_prompt)
        cached_global_ids = global_input_tensor[0].tolist()

        prompt_len = len(prompts)

        for i, prompt in enumerate(prompts):
            start_time: float = time.time()
            current_prompt = self.prompter.prepare_user_query(prompt)
            print(f"[{i+1}/{prompt_len}] Processing query: '{prompt.prompt}'",
                  end="")

            input_tensor = self.llm.encode(current_prompt)
            user_input_ids = input_tensor[0].tolist()

            self.input_ids = cached_global_ids + user_input_ids

            # print("Output: ", end="", flush=True)
            decoder.reset_state()

            i = 0
            while decoder.current_state != "DONE":
                # print(f"#{i} : '{decoder.generated_text}'")
                if self.add_tokens_to_context_if_possible(decoder):
                    continue

                raw_logits = np.array(
                    self.llm.get_logits_from_input_ids(self.input_ids),
                    dtype=np.float32
                )
                # Flatten the tensor if the model returns a sequence of logits.
                # We only care about the very last token's probabilities.
                while len(raw_logits.shape) > 1:
                    raw_logits = raw_logits[-1]

                filtered_logits_np = decoder.filter_logits(raw_logits)

                next_token_id = int(np.argmax(filtered_logits_np))
                self.input_ids.append(next_token_id)

                new_word = self.llm.decode([next_token_id])
                decoder.update_state(new_word)
                i += 1
                # print(new_word, end="", flush=True)
            # print(f"\nresult : '{decoder.generated_text}'")
            # print("\n" + "="*50 + "\n")
            tmp = format_final_result(prompt, decoder.generated_text)
            result.append(tmp)
            end_time: float = time.time()
            exec_time: float = end_time - start_time
            print(f"\033[92m   [OK]\033[0m ({exec_time:.5f} seconds)")

        return result
