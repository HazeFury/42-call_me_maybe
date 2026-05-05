import numpy as np
from src.core.vocab_manager import VocabManager
from src.utils.validators import FunctionValidator


class ConstrainedDecoder:
    """
    Enforces JSON schema compliance during LLM generation using a
    Finite State Machine.
    """

    def __init__(
            self,
            vocab_manager: VocabManager,
            functions_defs: list[FunctionValidator]
            ):
        self.vocab_manager = vocab_manager
        self.functions_catalog = functions_defs

        # Adding the Ping-Pong states for parameters
        self.state_sequence = [
            "FUNCTION_NAME",
            "PARAMS_KEY",
            "PARAM_KEY",
            "PARAM_VALUE",
            "CLOSING_BRACE",
            "DONE"
        ]

        self.current_state_idx = 0
        self.state_buffer: str = ""
        self.generated_text: str = '{ "name": "'

        self.chosen_function: str | None = None
        self.params_queue: list[tuple[str, str]] = []
        self.current_param: tuple[str, str] | None = None

    # ==========================================================================
    # Properties & State Management
    # ==========================================================================

    @property
    def current_state(self) -> str:
        """Shortcut to get the current state."""
        return self.state_sequence[self.current_state_idx]

    def set_state(self, state_name: str) -> None:
        """Jump to a specific state by name."""
        self.current_state_idx = self.state_sequence.index(state_name)
        self.state_buffer = ""
        # print(f"[DEBUG] just switch to {self.current_state}")

    def go_to_next_state(self) -> None:
        """Go to the next sequential state."""
        if self.current_state_idx < len(self.state_sequence) - 1:
            self.current_state_idx += 1
            self.state_buffer = ""
            # print(f"[DEBUG] just switch to {self.current_state}")

    def reset_state(self) -> None:
        """Reset the state machine."""
        self.current_state_idx = 0
        self.state_buffer = ""
        self.generated_text = '{ "name": "'
        self.chosen_function = None
        self.params_queue.clear()
        self.current_param = None

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _load_function_parameters(self, func_name: str) -> None:
        """
        Finds the function in the catalog and loads its parameters into the
        queue. Assuming FunctionValidator has a 'parameters' dict with
        a 'properties' dict.
        """
        for func in self.functions_catalog:
            if func.name == func_name:

                for param_name, param_validator in func.parameters.items():

                    param_type = param_validator.type

                    self.params_queue.append((param_name, param_type))

                break

    # ==========================================================================
    # The Core Logic
    # ==========================================================================

    def update_state(self, new_token_string: str) -> None:
        """
        Updates the internal state of the machine based on the newly
        accepted token.
        """
        self.generated_text += new_token_string
        self.state_buffer += new_token_string

        if self.current_state == "FUNCTION_NAME":
            for func in self.functions_catalog:
                target = func.name + '"'
                if self.state_buffer.strip() == target:
                    self.chosen_function = func.name
                    self._load_function_parameters(func.name)
                    self.go_to_next_state()
                    break

        elif self.current_state == "PARAMS_KEY":
            self.go_to_next_state()

        elif self.current_state == "PARAM_KEY":
            self.go_to_next_state()  # Goes to PARAM_VALUE

        elif self.current_state == "PARAM_VALUE":
            # What is the character that ends this value?
            terminal_char = "," if self.params_queue else "}"

            # If the LLM generates the terminal character, we switch states
            if self.state_buffer.strip().endswith(terminal_char):
                self.current_param = None  # Clear current parameter

                if terminal_char == ",":
                    # Ping-pong back to PARAM_KEY for the next parameter
                    self.set_state("PARAM_KEY")
                else:
                    # The JSON is complete!
                    self.set_state("CLOSING_BRACE")

        elif self.current_state == "CLOSING_BRACE":
            self.go_to_next_state()

    def filter_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Evaluates next tokens using optimized pre-computed sets for O(1)
        matching on numbers and booleans, drastically reducing inference time.
        """
        valid_ids: list[int] = []

        if self.current_state == "FUNCTION_NAME":
            # For the function name, we still need to check all tokens,
            # but we use the pre-cleaned dictionary which is very fast.
            for token_id, token_str in \
                    self.vocab_manager.clean_id_to_token.items():
                simulated_buffer = (self.state_buffer + token_str).lstrip()

                is_valid_func = False
                for func in self.functions_catalog:
                    target = func.name + '"'
                    if target.startswith(simulated_buffer):
                        is_valid_func = True
                        break

                if is_valid_func:
                    valid_ids.append(token_id)

        elif self.current_state == "PARAM_VALUE":
            terminal_char = "," if self.params_queue else "}"
            wrong_terminal = "}" if terminal_char == "," else ","

            param_type = self.current_param[1] if self.current_param else \
                "string"

            # =================================================================
            # THE FAST PATH (Numbers & Booleans)
            # We ONLY iterate over the VIP lists, skipping 149,000+ tokens!
            # =================================================================
            if param_type in ["integer", "number"]:
                # Combine number tokens & stop tokens (commas, spaces, braces)
                candidate_ids = (self.vocab_manager.tokens_number |
                                 self.vocab_manager.tokens_stop)

                for token_id in candidate_ids:
                    token_str = self.vocab_manager.clean_id_to_token[token_id]
                    if wrong_terminal in token_str:
                        continue
                    valid_ids.append(token_id)

            elif param_type == "boolean":
                candidate_ids = (self.vocab_manager.tokens_boolean |
                                 self.vocab_manager.tokens_stop)

                for token_id in candidate_ids:
                    token_str = self.vocab_manager.clean_id_to_token[token_id]
                    if wrong_terminal in token_str:
                        continue
                    valid_ids.append(token_id)

            # =================================================================
            # THE STRING PATH
            # =================================================================
            elif param_type == "string":
                # Strings can contain anything, so we must check all tokens.
                for token_id, token_str in \
                        self.vocab_manager.clean_id_to_token.items():
                    simulated_buffer = (self.state_buffer + token_str).lstrip()

                    if wrong_terminal in token_str:
                        continue

                    if self.state_buffer.strip() == "":
                        if not token_str.lstrip().startswith('"') and \
                                token_str.strip() != "":
                            continue

                    if terminal_char in token_str:
                        if simulated_buffer.count('"') < 2:
                            continue

                    valid_ids.append(token_id)

        # =====================================================================
        # FAST NATIVE MASKING (Pure Python, No Numpy overhead)
        # =====================================================================
        mask = np.full(logits.shape, -1e11, dtype=np.float32)

        # 2. Vectorized assignment: copy all valid probabilities at once!
        # This is the true power of NumPy, replacing the Python 'for' loop.
        if valid_ids:
            mask[valid_ids] = logits[valid_ids]

        return mask
