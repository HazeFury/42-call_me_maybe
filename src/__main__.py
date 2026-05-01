import sys
import time
from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.generation_orchestrator import GenerationOrchestrator
from src.core.prompt_builder import PromptBuilder
from src.utils.parser import get_args
from src.utils.validators import format_and_cast_results
from src.utils.file_handler import export_json_to_file


def main() -> None:
    # ==========================  PARSING  =============================
    start_time: float = time.time()
    try:
        functions, prompts, output_path = get_args()

        if len(prompts) == 0:
            raise ValueError("Please enter at least one prompt to "
                             "start the program !")

    except Exception as e:
        print("[ERROR] An error occured during "
              f"parsing :\n => {e}\n")
        sys.exit(1)

    # =========================  GENERATION  =============================

    try:
        llm = Small_LLM_Model()

        prompter = PromptBuilder(functions)

        orchestrator = GenerationOrchestrator(llm, prompter)
        raw_json = orchestrator.run_generation(prompts, functions)
    except Exception as e:
        print("[ERROR] An error occured during initialization or "
              f"running the main loop of generation :\n => {e}\n")
        sys.exit(1)

    # =========================  EXPORTING  ==============================

    try:
        print("\n\n" + "#"*50 + "\n\n")
        print(raw_json)
        print("\n\n")

        type_checked_raw_json = format_and_cast_results(
            content=raw_json,
            function_defs=functions
            )

        export_json_to_file(
            content=type_checked_raw_json,
            path=output_path)

        end_time: float = time.time()
    except Exception as e:
        print("[ERROR] An error occured during the last check and "
              f"exporting to file :\n => {e}\n")
        sys.exit(1)
    else:
        print("[SUCCESS] Everything went well :) You will found your output in"
              f"following path :\n {output_path}")
        execution_time: float = end_time - start_time
        print(f"Processed all prompts in {execution_time:.5f} seconds")


if __name__ == "__main__":
    main()
