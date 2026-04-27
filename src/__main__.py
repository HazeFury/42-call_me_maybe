import sys
from llm_sdk import Small_LLM_Model  # type: ignore
from src.core.generation_orchestrator import GenerationOrchestrator
from src.core.prompt_builder import PromptBuilder
from src.utils.parser import get_args


def main() -> None:
    # ==========================  PARSING  =============================
    try:
        functions, prompts = get_args()

        if len(prompts) == 0:
            raise ValueError("Please enter at least one prompt to "
                             "start the program !")

    except Exception as e:
        print("[ERROR] An error occured during "
              f"parsing :\n {e}\n")
        sys.exit(1)

    # =========================  GENERATION  =============================

    try:
        llm = Small_LLM_Model()

        prompter = PromptBuilder(functions)

        orchestrator = GenerationOrchestrator(llm, prompter)
        orchestrator.run_generation(prompts)
    except Exception as e:
        print("[ERROR] An error occured during initialization or "
              f"running the main loop of generation :\n {e}\n")

    # ====================================================================


if __name__ == "__main__":
    main()
