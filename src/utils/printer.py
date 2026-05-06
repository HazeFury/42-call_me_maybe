class Printer:
    def introduction() -> None:
        print("\n\n" + "\033[105m=\033[0m"*101 + "\n" +
              "\033[105m=\033[0m"*40 +
              "    CALL ME MAYBE    " +
              "\033[105m=\033[0m"*40 + "\n" +
              "\033[105m=\033[0m"*101)

    def display_step(step: int) -> None:
        if step == 0:
            print("\n\033[104m[STARTING]\033[0m\n\n")
            return
        if step == 5:
            print("\n\033[104m[ENDING]\033[0m\n")
            return

        message: str = ""

        if step == 1:
            message = "Trying to read arguments"
        elif step == 2:
            message = "Instanciate the LLM and other needed tools"
        elif step == 3:
            message = "Starting the generation process." \
                      "(Be patient, this may take a few minutes)\n"
        elif step == 4:
            message = "Verifying result and exporting it to json file"
        else:
            return

        print(f"STEP {step} : {message}")

    def step_successed(step) -> None:
        if step < 1 or step > 4:
            return

        print(f"\033[92m[SUCCESS]\033[35m STEP {step}\033[0m completed without"
              " error")
        print("\n\033[34m===========================================\033[0m\n")

    def successful_end(
            output_path: str,
            exec_time: float,
            total_prompt: int
            ) -> None:

        time_in_minutes = f"{int(exec_time // 60)}m{int(exec_time % 60)}"

        print("\033[92m[SUCCESS]\033[0m All jobs completed successfully :)\n")
        print(f"Total prompts processed : \033[35m{total_prompt}\033[0m")
        print(f"Output file path : \033[35m{output_path}\033[0m")
        print(f"Time of process : \033[35m{exec_time:.5f} seconds "
              f"\033[34m({time_in_minutes})\033[0m")

    def error(text: str) -> None:
        print(f"\033[91m[ERROR]\033[0m {text}\n")
