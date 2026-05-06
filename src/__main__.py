import sys
import time
from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.generation_orchestrator import GenerationOrchestrator
from src.core.prompt_builder import PromptBuilder
from src.utils.parser import get_args
from src.utils.validators import format_and_cast_results
from src.utils.file_handler import export_json_to_file
from src.utils.printer import Printer


def main() -> None:
    # ==========================  PARSING  =============================
    printer = Printer

    printer.introduction()
    printer.display_step(step=0)
    printer.display_step(step=1)

    try:
        functions, prompts, output_path = get_args()
        if len(prompts) == 0:
            raise ValueError("Please enter at least one prompt to "
                             "start the program !")

    except Exception as e:
        printer.error(f"An error occured during parsing :\n => {e}")
        sys.exit(1)
    else:
        printer.step_succeeded(step=1)

    # # =========================  GENERATION  =============================

    try:
        printer.display_step(step=2)

        llm = Small_LLM_Model()
        prompter = PromptBuilder(llm, functions)
        orchestrator = GenerationOrchestrator(llm, prompter)

        printer.step_succeeded(step=2)
        printer.display_step(step=3)

        start_time: float = time.time()

        raw_json = orchestrator.run_generation(prompts, functions)

        end_time: float = time.time()

    except Exception as e:
        printer.error("An error occured during the run of the"
                      f" main loop of generation :\n => {e}")
        sys.exit(1)
    else:
        printer.step_succeeded(step=3)

    # # =========================  EXPORTING  ==============================

    try:
        printer.display_step(step=4)

        type_checked_raw_json = format_and_cast_results(
            content=raw_json,
            function_defs=functions
            )

        export_json_to_file(
            content=type_checked_raw_json,
            path=output_path)

        printer.step_succeeded(step=4)

    except Exception as e:
        printer.error("An error occured during the last check"
                      f" and/or export to file process :\n => {e}\n")
        sys.exit(1)
    else:
        exec_time: float = end_time - start_time
        printer.successful_end(
            output_path=output_path,
            exec_time=exec_time,
            total_prompt=len(prompts)
            )
        printer.display_step(5)


if __name__ == "__main__":
    main()
