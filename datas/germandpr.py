import datasets
from TOKENS import BOT, PROMPTER, END
import random


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

        ctx_string = ""
        for context in ctx:
            ctx_string += f"{context}\n"

        PROMPT = f"""{PROMPTER}Extrahiere die Antwort auf die Frage aus dem Text.
Frage: {question}
Text: {ctx_string}{END}{BOT}{answer}{END}"""

        all_rows.append(PROMPT)
        all_labels.append("closed_qa")
        from_ds.append("deepset/germandpr")

    return all_rows, all_labels, from_ds
