import json
from pathlib import Path
from pydantic import TypeAdapter
from typing import Any
from collections.abc import Sequence
from src.utils.validators import FunctionValidator, PromptValidator


def parse_file_to_json(file_path: str) -> Any:
    """
    Open and read a JSON file and return content as JSON format.

    Error cases:
    - missing file
    - unsupported file type
    - permission error,
    - empty file.
    - invalid JSON
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing File: '{path.name}' not found")

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported file type: '{path.suffix}'. "
            f"Please provide a .json file"
            )

    if path.is_dir():
        raise IsADirectoryError("Expected a file, but found a directory")

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            content = f.read().strip()
    except PermissionError:
        raise PermissionError(
            f"You don't have permisssion for this file: {file_path}"
            )

    if not content:
        raise ValueError(f"File is empty: '{path.name}'")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{path.name}': "
                         f"Line {e.lineno}, column {e.colno}.")


def get_items_from_json(
        file_path: str, item_type: str
        ) -> Sequence[FunctionValidator | PromptValidator]:
    """
    Extract and validates content using Pydantic model validators.

    Args:
    - file_path = path of the JSON file you want to open.
    - item_type = type of the data you want to validate.

    Returns a list of validated Pydantic objects
    """

    if item_type not in ("func", "prompt"):
        raise ValueError("Wrong value for 'item_type' parameter. "
                         "Expected value is 'func' or 'prompt'")

    raw_json = parse_file_to_json(file_path)

    if item_type == "func":
        adapter = TypeAdapter(list[FunctionValidator])
    else:
        adapter = TypeAdapter(list[PromptValidator])

    items = adapter.validate_python(raw_json)

    if item_type == "func" and not items:
        raise ValueError(
            f"No functions found in : {file_path}\n"
            "You must have at least one function available"
            )

    return items
