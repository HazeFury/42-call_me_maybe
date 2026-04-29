import argparse
from typing import cast
from src.utils.validators import FunctionValidator, PromptValidator
from src.utils.file_handler import get_items_from_json


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Constrained Decoding LLM Tool"
    )

    parser.add_argument(
        "-f",
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definition file"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to user prompts file"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output file or directory"
    )

    args = parser.parse_args()
    return args


def get_args() -> tuple[list[FunctionValidator], list[PromptValidator], str]:
    args = parse_arguments()

    validated_functions = cast(
        list[FunctionValidator],
        get_items_from_json(
            file_path=args.functions_definition,
            item_type="func"
        )
    )

    validated_prompts = cast(
        list[PromptValidator],
        get_items_from_json(
            file_path=args.input,
            item_type="prompt"
        )
    )

    output_path: str = args.output

    return (validated_functions, validated_prompts, output_path)
