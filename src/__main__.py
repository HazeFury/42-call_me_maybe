from src.utils.file_to_json import get_items_from_json
from src.utils.parser import parse_arguments


def main() -> None:
    try:
        args = parse_arguments()

        print("received paths :\n"
              f"{args.functions_definition}\n"
              f"{args.input}\n"
              f"{args.output}\n\n")

        validated_functions = get_items_from_json(
            file_path=args.functions_definition,
            item_type="func"
        )
        functions = [f.model_dump() for f in validated_functions]

        validated_prompts = get_items_from_json(
            file_path=args.input,
            item_type="prompt"
        )
        prompts = [f.model_dump() for f in validated_prompts]

        print(functions)
        print("\n" + "="*40 + "\n")
        print(prompts)
        print("\n" + "="*40 + "\n")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
