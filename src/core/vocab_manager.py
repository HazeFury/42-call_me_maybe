import json


class VocabManager:
    """
    Loads and manages the model's vocabulary for fast token-to-string lookups.
    Optimized to pre-clean tokens on initialization.
    """

    def __init__(self, vocab_path: str):
        # We will store the CLEANED strings directly
        self.clean_id_to_token: dict[int, str] = {}

        try:
            with open(vocab_path, 'r', encoding="utf-8") as vocab_file:
                vocab: dict[str, int] = json.load(vocab_file)

                # We invert the dictionary AND clean the strings immediately
                for raw_string, token_id in vocab.items():
                    clean_string = raw_string.replace("Ġ", " ")
                    self.clean_id_to_token[token_id] = clean_string

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{vocab_path}': "
                             f"Line {e.lineno}, column {e.colno}.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_path}"
                )

    def get_token_string(self, token_id: int) -> str | None:
        """
        Returns the human-readable, cleaned string for a given token ID.
        Returns None if the token_id is not in the vocabulary.
        """
        # This is now an instant O(1) lookup with NO string manipulation!
        return self.clean_id_to_token.get(token_id)
