import sys
from llm_sdk import Small_LLM_Model
from src.core.generation_orchestrator import GenerationOrchestrator
from src.utils.parser import get_args


def main() -> None:
    try:
        functions, prompts = get_args()

        if len(prompts) == 0:
            raise ValueError("Please enter at least one prompt to "
                             "start the program !")

    except Exception as e:
        print("[ERROR] An error occured during "
              f"parsing :\n {e}")
        sys.exit(1)

    # ==================================================================

    try:
        llm = Small_LLM_Model()
        orchestrator = GenerationOrchestrator(llm)
        orchestrator.run_generation(functions, prompts)
    except Exception as e:
        print("[ERROR] An error occured during initialization or "
              f"running the main loop of generation :\n {e}")


if __name__ == "__main__":
    main()
