import json


class VocabManager:
    """
    Loads and manages the model's vocabulary for fast token-to-string lookups.
    """

    def __init__(self, vocab_path: str):
        self._id_to_token: dict[int, str] = {}

        try:
            with open(vocab_path, 'r', encoding="utf-8") as vocab_file:
                # json.load reads directly from the file object
                vocab: dict[str, int] = json.load(vocab_file)

                self._id_to_token = {
                    value: key for key, value in vocab.items()
                    }

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{vocab_path}': "
                             f"Line {e.lineno}, column {e.colno}.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_path}"
                )

    def _clean_token_string(self, raw_token: str) -> str:
        """
        Replaces special tokenizer characters with standard equivalents.
        For Qwen/BPE tokenizers, the 'Ġ' character represents a space.
        """
        return raw_token.replace("Ġ", " ")

    def get_token_string(self, token_id: int) -> str:
        """
        Returns the human-readable, cleaned string for a given token ID.
        Raises a KeyError if the token_id is not in the vocabulary.
        """
        if token_id not in self._id_to_token:
            raise KeyError(f"Token ID {token_id} not found in vocabulary.")

        raw_string = self._id_to_token[token_id]

        # We chain our cleaning function before returning the final text
        return self._clean_token_string(raw_string)
