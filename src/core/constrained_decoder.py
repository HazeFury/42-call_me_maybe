import math
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
            "OPENING_BRACE",
            "NAME_KEY",
            "FUNCTION_NAME",
            "PARAMS_KEY",
            "PARAM_KEY",
            "PARAM_VALUE",
            "CLOSING_BRACE",
            "DONE"
        ]

        self.state_target = {
            "OPENING_BRACE": "{",
            "NAME_KEY": '"name": "',
            "PARAMS_KEY": ', "parameters": {',
            "CLOSING_BRACE": "}"
        }

        self.current_state_idx = 0
        self.state_buffer: str = ""
        self.generated_text: str = ""

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

    def go_to_next_state(self) -> None:
        """Go to the next sequential state."""
        if self.current_state_idx < len(self.state_sequence) - 1:
            self.current_state_idx += 1
            self.state_buffer = ""

    def reset_state(self) -> None:
        """Reset the state machine."""
        self.current_state_idx = 0
        self.state_buffer = ""
        self.generated_text = ""
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

        if self.current_state in self.state_target:
            target = self.state_target[self.current_state]
            if self.state_buffer.strip() == target:
                self.go_to_next_state()

        elif self.current_state == "FUNCTION_NAME":
            for func in self.functions_catalog:
                target = func.name + '"'
                if self.state_buffer.strip() == target:
                    self.chosen_function = func.name
                    self._load_function_parameters(func.name)
                    self.go_to_next_state()
                    break

        elif self.current_state == "PARAM_KEY":
            # If we don't have a current parameter, pop one from the queue
            if not self.current_param and self.params_queue:
                self.current_param = self.params_queue.pop(0)

            # If the queue was empty and we have no param, we jump to end!
            if not self.current_param:
                self.set_state("DONE")
                return

            # We expect the LLM to write: '"param_name": '
            target_key = f'"{self.current_param[0]}":'

            # Using replace to ignore spaces the LLM might add around the colon
            buffer_no_spaces = self.state_buffer.replace(" ", "").strip()

            if buffer_no_spaces == target_key:
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

    def filter_logits(self, logits: list[float]) -> list[float]:
        """
        Evaluates all possible next tokens against the current state.
        Sets the logit to -math.inf for any token that violates the rules.
        """
        for token_id in range(len(logits)):
            string = self.vocab_manager.get_token_string(token_id)

            if string is None:
                logits[token_id] = -math.inf
                continue

            simulated_buffer = (self.state_buffer + string).lstrip()

            if self.current_state in self.state_target:
                target = self.state_target[self.current_state]
                if self.current_state in ("OPENING_BRACE", "CLOSING_BRACE"):
                    if string.strip() != "" and string.strip() != target:
                        logits[token_id] = -math.inf
                else:
                    if not target.startswith(simulated_buffer):
                        logits[token_id] = -math.inf

            elif self.current_state == "FUNCTION_NAME":
                is_valid = False
                for func in self.functions_catalog:
                    target = func.name + '"'

                    if target.startswith(simulated_buffer):
                        is_valid = True
                        break
                if not is_valid:
                    logits[token_id] = -math.inf

            elif self.current_state == "PARAM_KEY":
                if self.current_param:
                    # We tolerate spaces, so we remove them for strict check
                    target = f'"{self.current_param[0]}":'
                    sim_no_spaces = simulated_buffer.replace(" ", "")

                    if not target.startswith(sim_no_spaces) and not \
                            sim_no_spaces.startswith(target):
                        logits[token_id] = -math.inf

            elif self.current_state == "PARAM_VALUE":
                # 1. Prevent early termination
                terminal_char = "," if self.params_queue else "}"
                wrong_terminal = "}" if terminal_char == "," else ","

                if wrong_terminal in string:
                    logits[token_id] = -math.inf
                    continue

                # 2. Basic Type Enforcement
                param_type = \
                    self.current_param[1] if self.current_param else "string"

                if param_type in ["integer", "number"]:
                    # If it's a number, only allow digits, spaces,
                    # and the correct terminal char
                    allowed_chars = "0123456789. " + terminal_char
                    if any(char not in allowed_chars for char in string):
                        logits[token_id] = -math.inf

                elif param_type == "string":
                    # It's a string. It must be wrapped in quotes.
                    # For a robust implementation, we would count quotes here.
                    # As a baseline: if we see the terminal char, ensure it
                    #  comes AFTER a quote
                    if terminal_char in string:
                        # E.g., if string is '",', it's valid. If it's
                        # just ',', it's missing the closing quote.
                        if '"' not in simulated_buffer:
                            logits[token_id] = -math.inf

        return logits
