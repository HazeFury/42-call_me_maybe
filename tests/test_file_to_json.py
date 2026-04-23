import os
import pytest
from pathlib import Path
from pydantic import ValidationError
from src.utils.file_to_json import parse_file_to_json, get_items_from_json

# ==========================================
# TESTS FOR parse_file_to_json
# ==========================================


def test_parse_file_success(tmp_path: Path) -> None:
    """Test basic case : valid json file correctly read."""
    test_file = tmp_path / "valid.json"
    test_file.write_text('{"hello": "world"}', encoding="utf-8")

    result = parse_file_to_json(str(test_file))

    assert result == {"hello": "world"}


def test_parse_file_missing(tmp_path: Path) -> None:
    """Crash test if file doesn´t exist."""
    missing_file = tmp_path / "ghost.json"

    with pytest.raises(FileNotFoundError, match="Missing File"):
        parse_file_to_json(str(missing_file))


def test_parse_file_wrong_extension(tmp_path: Path) -> None:
    """Crash test if file extension isn't.json."""
    bad_ext_file = tmp_path / "data.txt"
    bad_ext_file.write_text("juste du texte")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file_to_json(str(bad_ext_file))


def test_parse_file_is_directory(tmp_path: Path) -> None:
    """Crash test if the path represent a folder instead of a file."""
    fake_dir = tmp_path / "sneaky_dir.json"
    fake_dir.mkdir()

    with pytest.raises(IsADirectoryError):
        parse_file_to_json(str(fake_dir))


def test_parse_file_empty(tmp_path: Path) -> None:
    """Crash test if file is empty."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")

    with pytest.raises(ValueError, match="File is empty"):
        parse_file_to_json(str(empty_file))


def test_parse_file_invalid_json(tmp_path: Path) -> None:
    """Crash test if the file contains invalide json."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{ "bad_key": 42 ', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in"):
        parse_file_to_json(str(invalid_file))


def test_parse_file_permission_error(tmp_path: Path) -> None:
    """Crash test if the file doesn't have read rights."""
    locked_file = tmp_path / "locked.json"
    locked_file.write_text('{"secret": "data"}')

    os.chmod(locked_file, 0o000)

    with pytest.raises(PermissionError):
        parse_file_to_json(str(locked_file))

    os.chmod(locked_file, 0o777)


# ==========================================
# TESTS POUR get_items_from_json
# ==========================================

def test_get_items_invalid_type(tmp_path: Path) -> None:
    """Test that the arguement 'item_type' is strictly controled."""
    test_file = tmp_path / "test.json"
    test_file.write_text('[]')

    with pytest.raises(
        ValueError, match="Wrong value for 'item_type' parameter"
            ):
        get_items_from_json(str(test_file), "pizza")


def test_get_items_functions_empty_list(tmp_path: Path) -> None:
    """Test if an empty list of function raise an error."""
    test_file = tmp_path / "empty_funcs.json"
    test_file.write_text('[]')

    with pytest.raises(ValueError, match="No functions found in"):
        get_items_from_json(str(test_file), "func")


def test_get_items_prompts_empty_list_allowed(tmp_path: Path) -> None:
    """Test if an empty list of prompt is authorized."""
    test_file = tmp_path / "empty_prompts.json"
    test_file.write_text('[]')

    result = get_items_from_json(str(test_file), "prompt")
    assert result == []


def test_get_items_validation_error(tmp_path: Path) -> None:
    """Test if Pydantic raise an error when schema is not respected."""
    bad_data = """
    [
        {
            "name": "fn_test",
            "description": "A good function",
            "parameters": {},
            "returns": {"type": "not_a_valid_type"}
        }
    ]
    """
    test_file = tmp_path / "bad_schema.json"
    test_file.write_text(bad_data)

    with pytest.raises(ValidationError):
        get_items_from_json(str(test_file), "func")


def test_get_items_success(tmp_path: Path) -> None:
    """
    Test if a perfect JSON is validated and transformed in a Pydantic object.
    """
    good_prompt = """
    [
        {"prompt": "What is the sum of 2 and 3?"}
    ]
    """
    test_file = tmp_path / "good_prompt.json"
    test_file.write_text(good_prompt)

    result = get_items_from_json(str(test_file), "prompt")

    assert len(result) == 1
    assert result[0].prompt == "What is the sum of 2 and 3?"
