from llm_sdk import Small_LLM_Model
from src.core.prompt_builder import PromptBuilder
from src.utils.validators import ResultValidator


class GenerationOrchestrator:
    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm
        self.prompter: PromptBuilder | None = None
        self._cache: dict[str, ResultValidator] = {}

    def run_generation(self, functions: str, prompts: str) -> None:

        self.prompter = PromptBuilder(functions)

        for prompt in prompts:

            current_prompt = self.prompter.build_prompt(prompt)
            print(current_prompt)

            # 1.On transforme ton texte en une suite de nombres (les Input IDs)
            # encode() renvoie un tenseur 2D,
            # on le convertit en liste 1D standard
            # input_tensor = self.llm.encode(current_prompt)
            # input_ids = input_tensor[0].tolist()
            # print(input_ids)
            # break
            # On va générer un maximum de 40 tokens
            # pour ne pas qu'il parle à l'infini
            # max_tokens = 40

            # for _ in range(max_tokens):
            #     # 2. On demande au modèle : "Étant donné cet historique,
            #     # quels sont les scores du prochain token ?"
            #     logits = self.llm.get_logits_from_input_ids(input_ids)

            #     # 3. Greedy Decoding : On choisit bêtement le token avec
            #     #  le score de probabilité le plus haut
            #     next_token_id = logits.index(max(logits))

            #     # 4. On ajoute ce nouveau token à notre historique pour
            #     # la prochaine itération de la boucle
            #     input_ids.append(next_token_id)

            #     # 5. On décode juste ce nouveau token pour te
            #     # l'afficher en direct dans le terminal
            #     new_word = self.llm.decode([next_token_id])
            #     print(new_word, end="", flush=True)

            print("\n\n" + "-"*40 + "\n")
