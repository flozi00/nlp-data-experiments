import datasets, random


def mtop_domain():
    prompt = []
    chosen = []
    rejected = []
    ds = datasets.load_dataset("mteb/mtop_domain", "de", split="train")

    possible_choices = ds.unique("label_text")

    SYS_PROMPTS = [
        f"Bestimme die Kategorie, welche am besten für den gegebenen Text passt. Du kannst folgende Kategorien wählen: {possible_choices}.",
        f"Klassifiziere den Text in eine der Kategorien: {possible_choices}.",
        f"Als Kategorisierungsassistent musst du den Text in eine der Kategorien einordnen. Du kannst folgende Kategorien wählen: {possible_choices}.",
    ]

    for entry in ds:
        tweet = entry["text"]
        sentiment = entry["label_text"]

        _PROMPT = f"""{random.choice(SYS_PROMPTS)}

{tweet}"""

        for x in possible_choices:
            if x == sentiment:
                continue
            else:
                prompt.append(_PROMPT)
                chosen.append(sentiment)
                rejected.append(x)

    return prompt, chosen, rejected
