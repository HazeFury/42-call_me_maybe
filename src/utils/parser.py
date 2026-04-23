import argparse


def parse_arguments():

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
