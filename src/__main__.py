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
    print("\033[34m=================================================\n"
          "===============    \033[35mCALL ME MAYBE\033[34m    =============\n"
          "=================================================\033[0m\n\n")
    print("[STARTING]\n\nSTEP 1 : Trying to read arguments")

    try:
        functions, prompts, output_path = get_args()
        if len(prompts) == 0:
            raise ValueError("Please enter at least one prompt to "
                             "start the program !")

    except Exception as e:
        print("\033[91m[ERROR]\033[0m An error occured during "
              f"parsing :\n => {e}\n")
        sys.exit(1)
    else:
        print("\033[92m[SUCCESS]\033[0m STEP 1 completed without error")
        print("\n\033[34m===========================================\033[0m\n")

    # # =========================  GENERATION  =============================

    try:
        print("STEP 2 : Instanciate the LLM and other needed tools")
        llm = Small_LLM_Model()

        prompter = PromptBuilder(llm, functions)

        orchestrator = GenerationOrchestrator(llm, prompter)
        print("\033[92m[SUCCESS]\033[0m STEP 2 completed without error")
        print("\n\033[34m===========================================\033[0m\n")
        print(f"STEP 3 : Starting the process on {len(prompts)} prompts."
              "(Be patient, this may take a few minutes)\n")

        start_time: float = time.time()
        raw_json = orchestrator.run_generation(prompts, functions)
        end_time: float = time.time()
    except Exception as e:
        print("\033[91m[ERROR]\033[0m An error occured during the run of the"
              f" main loop of generation :\n => {e}\n")
        sys.exit(1)
    else:
        print("\033[92m[SUCCESS]\033[0m STEP 3 completed without error")
        print("\n\033[34m===========================================\033[0m\n")

    # # =========================  EXPORTING  ==============================

    try:
        print("STEP 4 : Verifying result and exporting it to json file")
        type_checked_raw_json = format_and_cast_results(
            content=raw_json,
            function_defs=functions
            )

        export_json_to_file(
            content=type_checked_raw_json,
            path=output_path)

        print("\033[92m[SUCCESS]\033[0m STEP 4 completed without error")
        print("\n\033[34m===========================================\033[0m\n")

    except Exception as e:
        print("\033[91m[ERROR]\033[0m An error occured during the last check"
              f" and/or export to file process :\n => {e}\n")
        sys.exit(1)
    else:
        print("\033[92m[SUCCESS]\033[0m All jobs completed successfully :)\n"
              f"Output file path : \033[35m{output_path}\033[0m")
        exec_time: float = end_time - start_time
        time_in_minutes = f"{int(exec_time // 60)}m{int(exec_time % 60)}"
        print(f"Time of process : \033[35m{exec_time:.5f} seconds "
              f"({time_in_minutes})\033[0m\n")
        print("[ENDING]")


if __name__ == "__main__":
    main()
