_This project has been created as part of the 42 curriculum by marberge._

<div align="center">
<br>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQPzuYKu7n0cWUYa5Kbg0_LrlEQAIURWeo9A&s" alt="42 Logo" width="400" />

  <br>
</div>

# Call Me Maybe - LLM Function Calling

![Language](https://img.shields.io/badge/Language-python-blue)

<!-- ![Grade](https://img.shields.io/badge/Grade-125%2F100-brightgreen)
![Tag](https://img.shields.io/badge/TOCHANGE-grey) -->

## I. Description

### ✳️ Goal

Create a function calling tool that translates natural language
prompts into structured function calls. 

### ✳️ Overview

To edit

## II. Instructions

### Prerequisites
Before using this template, ensure you have the following installed on your system:
- **Python 3.10+**
- **uv 0.10.12+**

### Quick Start
To set up the environment and run the project for the first time, simply use the following command in your terminal:

	make run

### Makefile Commands Reference
This project is fully automated using Make. Here is the complete list of available commands to manage the project lifecycle:

**Installation & Setup**
- ```make install``` (or **make all**): Initializes the virtual environment (.venv) and synchronizes all dependencies using uv.
- ```make setup```: Checks your Python version and presence of the uv package manager. Exit if both check are not valid.

**Execution & Debugging**
- ```make run```: Executes the main entry point (src/main.py) inside the isolated virtual environment.
- ```make debug```: Launches the project using the Python Debugger (pdb), allowing you to step through your code line by line.

**Quality & Testing**
- ```make lint```: Runs flake8 for style checking and mypy for static type checking to ensure code quality.
- ```make lint-strict```: Runs the linters but enforces strict typing rules with mypy.
- ```make test```: Runs the entire test suite using pytest.
- ```make test-file FILE=path/to/test.py```: Runs a specific test file. Replace the FILE variable with your target.

**Building & Cleaning**
- ```make build```: Packages the project into distributable files inside a dist/ directory.
- ```make clean```: Removes all temporary files, such as __pycache__ folders and linter caches.
- ```make fclean```: Performs a deep clean. It executes the clean rule and also removes the virtual environment and build files.
- ```make re```: Rebuilds the project from scratch by running fclean followed by all.

### Manual Execution via uv

**1. Install dependencies**
```Bash
uv run python -m src
```

**2. Default execution**
```Bash
uv run python -m src
```

**3. Execution with custom arguments**
```Bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```


***

## III. Resources

### ✳️ References
To build this project, several key concepts were explored:

* Logit Masking & Constrained Decoding: https://huggingface.co/docs

* State Machine: https://fr.wikipedia.org/wiki/Automate_fini

* LLM et SLM : 

	https://www.ibm.com/fr-fr/think/topics/small-language-models

	https://blog.stephane-robert.info/docs/developper/programmation/python/llm/

* Qwen3-0.6B: https://qwen.ai/blog?id=qwen3

* argparse: tutoriel: https://docs.python.org/fr/3/howto/argparse.html

* uv: https://www.datacamp.com/fr/tutorial/python-uv?dc_referrer=https%3A%2F%2Fwww.google.com%2F

* numpy : https://numpy.org/doc/stable/

### ✳️ AI Usage
Strategic Planning: AI was used at the beginning of the project to brainstorm the modular decomposition of the pipeline (Separation of ConstrainedDecoder vs PromptProcessor).

Assistance in writing complex Regular Expressions for the logit masks.

Help with debugging specific numpy slicing operations and mypy type-hinting errors.

Testing: AI helped in creating the pytest structure and identifying edge cases for the function_calling_edge_cases.json file.

Documentation: Assistance in translating this README.md.


## Algorithm Explanation

The core of this constrained decoding engine is built around a **Finite State Machine (FSM)** combined with **Logit Masking**. Since Large Language Models (LLMs) operate by predicting the probabilities (logits) of the next token from a vocabulary of over 150,000 possibilities, our algorithm intercepts these probabilities before the final selection is made.

The FSM (`ConstrainedDecoder`) tracks the exact structural expectation of the JSON output. It transitions through specific states: `FUNCTION_NAME`, `PARAMS_KEY`, `PARAM_KEY`, `PARAM_VALUE`, and `CLOSING_BRACE`. 

Depending on the active state, the algorithm dynamically filters the vocabulary:
1. **Vocabulary Pre-computation:** At initialization, the `VocabManager` scans the entire vocabulary using Regular Expressions to categorize token IDs into specific sets (e.g., numbers, booleans, terminators).
2. **Fast-Path Masking:** When the FSM is in the `PARAM_VALUE` state and expects an `integer` or a `number`, the algorithm skips iterating over the 150,000 tokens. Instead, it performs an $O(1)$ union of the pre-computed sets (valid values + stop tokens) to identify the allowed token IDs.
3. **Vectorized Penalization:** A Numpy array is created and filled with `-1e11` (negative infinity). The original probabilities of the allowed tokens are copied into this mask using Advanced Indexing. The model is therefore forced to pick the most logical token *only* among the strictly valid options.

## Design Decisions

Several critical architectural choices were made to ensure robustness, speed, and modularity:

- **FSM over Procedural Parsing:** Instead of hardcoding a sequential extraction (forcing the name, then forcing the parameters one by one), the Finite State Machine approach allows the system to adapt to complex, nested, or variable JSON schemas dynamically. 
- **Numpy Vectorization and Memory Management:** Standard Python loops over 150,000 elements severely bottleneck inference speed. By enforcing `np.float32` arrays and using pure Numpy vectorization (`mask[valid_ids] = logits[valid_ids]`), we drastically reduced CPU cache misses and RAM overhead.
- **Pre-computed Token Sets (VIP Lists):** String manipulation (`startswith`, `strip`) during the generation loop is expensive. Moving the regex categorization to the script's initialization phase shifted the time complexity from $O(V)$ (where V is the vocabulary size) to $O(C)$ (where C is the number of valid candidates), saving massive amounts of computation time per token.
- **Prompt Caching:** An $O(1)$ dictionary-based caching system was implemented in the `GenerationOrchestrator`. If an identical prompt is detected, the LLM inference is bypassed entirely, returning the exact casted results from previous runs.
- **Strict Type Casting Fallback:** A dedicated casting function acts as a final safety net, ensuring that even if the FSM forces valid tokens, the final Python output strictly adheres to the requested types (e.g., converting `"true"` to a Python boolean `True`, or `"42"` to a float `42.0`).

## Performance Analysis

- **Accuracy & Reliability:** The FSM guarantees 100% compliance with the requested JSON schema. By forcing structural constraints at the logit level, the risk of hallucinations breaking the JSON format (missing commas, unclosed brackets, wrong types) is virtually eliminated.
- **Speed:** Inference speed was the primary bottleneck. Initial naive Python implementations took over 3 minutes for a batch of 11 prompts. By shifting the heavy lifting to Numpy, flattening multidimensional tensors securely (`raw_logits[-1]`), and relying on pre-computed token sets, the processing time was optimized to the physical limits of sequential CPU processing without using external industry-standard frameworks (like vLLM).
- **Caching Impact:** For repeated queries, the prompt caching drops the execution time from several seconds per prompt to near $0.00$ seconds, providing massive efficiency gains on repetitive datasets.

## Challenges Faced

1. **The "Conversion Tax" (Serialization Overhead):** 
Initially, translating Python lists to Numpy arrays and iterating over them natively negated the performance benefits of Numpy. The solution required abandoning Python `for` loops during the masking phase and fully utilizing Numpy's advanced indexing feature.

2. **Tensor Dimensionality Mismatches:** 
The LLM SDK occasionally returned multidimensional logit arrays `(Sequence, Vocab)` instead of a flat 1D vector `(Vocab)`. This caused unpredictable `IndexError` crashes. Implementing a dimension-stripping loop (`while len(logits.shape) > 1`) ensured stability regardless of the context size.

3. **Tokenizer Spacing Quirks:** 
Dealing with standard BPE tokenizers meant handling spaces explicitly (like the `Ġ` character). Trimming and matching tokens contextually inside the FSM without breaking the strict validation logic required careful tuning of the FSM's `state_buffer`.

4. **Strict Typing with Mypy:**
Enforcing strong typing architecture (to prevent runtime errors) required extensive use of Type Narrowing, especially around FSM states and parameter queues (e.g., ensuring `current_param` was not `None` before sequence indexing).

## Testing Strategy

To validate the implementation, a comprehensive suite of unit tests and integration tests was established:
- **Diverse Input Scenarios:** The engine was fed with a variety of prompts testing different parameter types: strings, floats, integers, and booleans.
- **Edge-Case Handling:** Specific tests were created to challenge the FSM (e.g., empty strings, regex patterns containing special characters, scientific notation for numbers).
- **Cache Validation:** Duplicate prompts were intentionally injected to verify that the FSM was skipped and the exact same object structure was instantly returned.
- **Static Analysis:** The entire codebase was subjected to strict static typing checks using `Mypy` and style enforcement via `Flake8` to comply with standard Python development norms (mirroring the rigor of the 42 Norminette).

## Example Usage

To run the constrained decoding engine, you can execute the main module. Make sure your environment is properly set up with the required dependencies (Numpy).

	# Run the main generation pipeline
	python3 -m src

The pipeline will automatically load the prompts from the input directory, process them through the `GenerationOrchestrator`, and output the correctly formatted JSON file. 

Example of an expected JSON output generated by the engine:

	[
	    {
	        "prompt": "What is the sum of 2 and 3?",
	        "name": "fn_add_numbers",
	        "parameters": {
	            "a": 2.0,
	            "b": 3.0
	        }
	    },
	    {
	        "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
	        "name": "fn_substitute_string_with_regex",
	        "parameters": {
	            "source_string": "The cat sat on the mat with another cat",
	            "regex": "cat",
	            "replacement": "dog"
	        }
	    }
	]
