from src.utils.file_to_json import get_items_from_json


def main() -> None:
    try:
        validated_functions = get_items_from_json(
            file_path="data/input/function_calling_tests.json",
            item_type="prompt"
        )
        functions = [f.model_dump() for f in validated_functions]
        print(functions)
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
