import datasets, random

SYS_PROMPTS = [
    "Du bist ein Sentiment-Analyst und musst das Sentiment eines Tweets bestimmen. Du kannst aus 3 Kategorien wählen: positiv, negativ oder neutral. 0 steht für negativ, 1 für neutral und 2 für positiv.",
    "Klassifiziere den Sentiment eines Tweets in eine der 3 Kategorien: positiv, negativ oder neutral. 0 steht für negativ, 1 für neutral und 2 für positiv.",
    "Als Stimmungsanalyst musst du das Sentiment eines Tweets bestimmen. Du kannst aus 3 Kategorien wählen: positiv, negativ oder neutral. Beginnend bei 0 für negativ, 1 für neutral und 2 für positiv.",
]


def sentiment_mteb_dpo():
    prompt = []
    chosen = []
    rejected = []
    ds = datasets.load_dataset(
        "mteb/tweet_sentiment_multilingual", "german", split="train"
    )

    for entry in ds:
        tweet = entry["text"]
        sentiment = entry["label"]

        _PROMPT = f"""{random.choice(SYS_PROMPTS)}

{tweet}"""

        for x in ["0", "1", "2"]:
            if x == sentiment:
                continue
            else:
                prompt.append(_PROMPT)
                chosen.append(sentiment)
                rejected.append(x)

    return prompt, chosen, rejected
