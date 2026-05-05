import json
import re


class VocabManager:
    """
    Loads and manages the model's vocabulary for fast token-to-string lookups.
    Optimized with pre-computed token sets for fast constrained decoding.
    """

    def __init__(self, vocab_path: str):
        self.clean_id_to_token: dict[int, str] = {}

        # Pre-computed sets for blazing fast O(1) lookups during generation
        self.tokens_number: set[int] = set()
        self.tokens_boolean: set[int] = set()
        self.tokens_stop: set[int] = set()

        try:
            with open(vocab_path, 'r', encoding="utf-8") as vocab_file:
                vocab: dict[str, int] = json.load(vocab_file)

                # 1. Clean and invert the dictionary IMMEDIATELY
                for raw_string, token_id in vocab.items():
                    clean_string = raw_string.replace("Ġ", " ")
                    self.clean_id_to_token[token_id] = clean_string

            # 2. Build the VIP lists (Sets) once and for all!
            self._build_optimized_sets()

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{vocab_path}': "
                             f"Line {e.lineno}, column {e.colno}.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_path}"
                )

    def _build_optimized_sets(self) -> None:
        """
        Iterates over the cleaned vocabulary exactly once to categorize
        tokens into sets (numbers, booleans, terminators) using Regex.
        """
        # Regex patterns
        # Numbers: digits, dots, minus signs, e/E for scientific notation
        pattern_num = re.compile(r'^[0-9.\-eE]+$')
        # Booleans: strict True, False, true, false
        pattern_bool = re.compile(r'^(true|false|True|False)$')
        # Terminators: commas, closing braces/brackets, colons, whitespaces
        pattern_stop = re.compile(r'^[,\}\]:\s\n\t]+$')

        for token_id, token_str in self.clean_id_to_token.items():
            # Strip leading/trailing spaces for accurate type matching
            stripped_str = token_str.strip()

            # 1. Check if it's purely a stopping character/whitespace
            if re.match(pattern_stop, token_str):
                self.tokens_stop.add(token_id)

            if not stripped_str:
                continue  # If it was just spaces, we move to the next token

            # 2. Check if it's a valid number token
            if re.match(pattern_num, stripped_str):
                self.tokens_number.add(token_id)

            # 3. Check if it's a valid boolean token
            if re.match(pattern_bool, stripped_str):
                self.tokens_boolean.add(token_id)

    def get_token_string(self, token_id: int) -> str | None:
        """
        Returns the pre-cleaned string for a given token ID.
        Returns None if the token_id is not in the vocabulary.
        """
        # This is now an instant O(1) lookup with NO string manipulation!
        return self.clean_id_to_token.get(token_id)
