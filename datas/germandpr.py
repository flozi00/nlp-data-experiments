import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
import random

SYSTEM_PROMPTS = [
    "Im folgenden stellt ein Nutzer sowohl eine Frage als auch einen Text bereit. Die Antwort auf die Frage ist im Text enthalten und wird vom Assistenten extrahiert.",
    "Gegeben ist eine Konversation zwischen einem Nutzer und einem Assistenten. Die Aufgabe des Assistenten ist es mittels des vom Nutzer gegebenen Textes seine Frage zu beantworten. Die Antwort ist ein Zitat aus dem Text.",
    "Du bist ein Assistent. Ein Nutzer stellt dir eine Frage und gibt dir einen Text. Die Antwort auf die Frage ist ein Zitat aus dem Text.",
    "Ein Assistent bekommt einen Text und eine Frage gestellt. Die Antwort auf die Frage ist ein Abschnitt aus dem Text.",
    "Im folgenden wird ein Text und eine Frage gestellt. Die Antwort auf die Frage ist ein Teil des Textes",
]


def germandpr():
    all_rows = []
    all_labels = []
    from_ds = []
    ds = datasets.load_dataset(
        "deepset/germandpr",
        split="train",
    )

    for entry in ds:
        question = entry["question"]
        positive_ctxs = entry["positive_ctxs"]
        hard_negative_ctxs = entry["hard_negative_ctxs"]
        negative_ctxs = entry["negative_ctxs"]
        answers = entry["answers"]
        answer = max(answers, key=len)
        ctx = []
        for positive_ctx in positive_ctxs["text"]:
            ctx.append(positive_ctx)
        for hard_negative_ctx in hard_negative_ctxs["text"]:
            ctx.append(hard_negative_ctx)
        for negative_ctx in negative_ctxs["text"]:
            ctx.append(negative_ctx)

        random.shuffle(ctx)

        ctx_string = "\n".join(ctx)

        PROMPT = f"""{SYSTEM}{random.choice(SYSTEM_PROMPTS)}{END}
{PROMPTER}Frage: {question}
Text: {ctx_string}{END}{BOT}{answer}{END}"""

        all_rows.append(PROMPT)
        all_labels.append("closed_qa")
        from_ds.append("deepset/germandpr")

    return all_rows, all_labels, from_ds
