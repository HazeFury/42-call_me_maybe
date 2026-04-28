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
        self.state_sequence = [
            "OPENING_BRACE",
            "NAME_KEY",
            "FUNCTION_NAME",
            "PARAMS_KEY",
            "PARAMS_DICT",
            "CLOSING_BRACE"
        ]
        self.state_target = {
            "OPENING_BRACE": "{",
            "NAME_KEY": '"name": "',
            "PARAMS_KEY": '", "parameters": {',
            "CLOSING_BRACE": "}"
        }
        self.current_state_idx = 0
        self.state_buffer: str = ""
        self.generated_text: str = ""

    # ==========================================================================

    @property
    def current_state(self) -> str:
        """Shortcut to get the current state ."""
        return self.state_sequence[self.current_state_idx]

    def go_to_next_state(self) -> None:
        """Go to the next state by incrementing index."""
        if self.current_state_idx < len(self.state_sequence) - 1:
            self.current_state_idx += 1
            self.state_buffer = ""
            print(f"\n--- [DEBUG-FSM] Transitioned to: {self.current_state}")

    def reset_state(self) -> None:
        """Reset the state machine."""
        self.current_state_idx = 0
        self.state_buffer = ""
        self.generated_text = ""

    # ==========================================================================

    def update_state(self, new_token_string: str) -> None:
        """
        Updates the internal state of the machine based on the newly accepted
        token. This is called ONLY ONCE per generation step, after the best
        token is chosen.
        """
        self.generated_text += new_token_string
        self.state_buffer += new_token_string

        if self.current_state in self.state_target.keys():

            target = self.state_target[self.current_state]

            if self.state_buffer.strip() == target:
                self.go_to_next_state()

        elif self.current_state == "GENERATING_FUNCTION_NAME":
            pass

        elif self.current_state == "GENERATING_PARAMS_DICT":
            pass

    # ==========================================================================

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

            if self.current_state in self.state_target.keys():

                target = self.state_target[self.current_state]

                if self.current_state in (
                    "OPENING_BRACE", "CLOSING_BRACE"
                ):
                    if string.strip() != "" and string.strip() != target:
                        logits[token_id] = -math.inf

                elif self.current_state in (
                    "NAME_KEY", "PARAMS_KEY"
                ):
                    simulated_buffer = (self.state_buffer + string).lstrip()

                    if not target.startswith(simulated_buffer) and not \
                            simulated_buffer.startswith(target):
                        logits[token_id] = -math.inf

            elif self.current_state == "GENERATING_FUNCTION_NAME":
                pass

            elif self.current_state == "GENERATING_PARAMS_DICT":
                pass

        return logits
