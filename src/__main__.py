from llm_sdk import Small_LLM_Model


def main() -> None:
    print("⏳ Chargement du modèle en mémoire "
          "(ça peut prendre quelques secondes)...")
    llm = Small_LLM_Model()
    print("✅ Modèle prêt ! (Tape 'q' pour quitter)\n")

    while True:
        prompt = input("👤 Toi : ")

        if prompt.strip().lower() == "q":
            break

        print("🤖 Qwen : ", end="", flush=True)

        # 1. On transforme ton texte en une suite de nombres (les Input IDs)
        # encode() renvoie un tenseur 2D, on le convertit en liste 1D standard
        input_tensor = llm.encode(prompt)
        input_ids = input_tensor[0].tolist()

        # On va générer un maximum de 40 tokens
        # pour ne pas qu'il parle à l'infini
        max_tokens = 40

        for _ in range(max_tokens):
            # 2. On demande au modèle : "Étant donné cet historique,
            # quels sont les scores du prochain token ?"
            logits = llm.get_logits_from_input_ids(input_ids)

            # 3. Greedy Decoding : On choisit bêtement le token avec
            #  le score de probabilité le plus haut
            next_token_id = logits.index(max(logits))

            # 4. On ajoute ce nouveau token à notre historique pour
            # la prochaine itération de la boucle
            input_ids.append(next_token_id)

            # 5. On décode juste ce nouveau token pour te
            # l'afficher en direct dans le terminal
            new_word = llm.decode([next_token_id])
            print(new_word, end="", flush=True)

        print("\n\n" + "-"*40 + "\n")


if __name__ == "__main__":
    main()
