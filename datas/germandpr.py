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

RAG_SYSTEM_PROMPTS = [
    "Du bist ein RAG Assistent. Ein Nutzer stellt dir eine Frage und gibt dir einen Text. Klassifiziere den Text als relevant oder irrelevant für die Frage.",
    "Im folgenden wird ein Text und eine Frage gestellt. Klassifiziere den Text als relevant oder irrelevant für die Frage.",
    "Ein RAG Assistent bekommt einen Text und eine Frage gestellt. Der Assistent soll den Text als relevant oder irrelevant für die Frage klassifizieren.",
    "Ein Text und eine Frage müssen beurteilt werden. Der Text ist relevant oder irrelevant für die Frage.",
    "Im folgenden wird ein Text und eine Frage ausgeführt. Der Text ist relevant oder irrelevant für die Frage.",
    "Ein Klassifizierer muss einen Text und eine Frage beurteilen. Der Text ist relevant oder irrelevant für die Frage.",
]


def germandpr() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []
    ds = datasets.load_dataset("deepset/germandpr", split="train")

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

        ctx = [f"Frage: {question}", f"Text: {ctx_string}"]
        random.shuffle(ctx)
        ctx_string = "\n\n".join(ctx)

        PROMPT = f"""{SYSTEM}{random.choice(SYSTEM_PROMPTS)}{END}
{PROMPTER}{ctx_string}{END}{BOT}{answer}{END}"""

        all_rows.append(PROMPT)
        all_labels.append("closed_qa")
        from_ds.append("deepset/germandpr")

    return all_rows, from_ds, all_labels


def germandpr_rag() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []
    ds = datasets.load_dataset("deepset/germandpr", split="train")

    for entry in ds:
        question = entry["question"]
        positive_ctxs = entry["positive_ctxs"]
        hard_negative_ctxs = entry["hard_negative_ctxs"]
        negative_ctxs = entry["negative_ctxs"]

        accepted_ctxs = []
        declined_ctxs = []

        for positive_ctx in positive_ctxs["text"]:
            accepted_ctxs.append(positive_ctx)
        for hard_negative_ctx in hard_negative_ctxs["text"]:
            declined_ctxs.append(hard_negative_ctx)
        for negative_ctx in negative_ctxs["text"]:
            declined_ctxs.append(negative_ctx)

        for ctx in accepted_ctxs:
            PROMPT = f"""{SYSTEM}{random.choice(RAG_SYSTEM_PROMPTS)}{END}
{PROMPTER}Passage: {ctx}\n\nFrage: {question}{END}{BOT}{"relevant"}{END}"""

        for ctx in declined_ctxs:
            PROMPT = f"""{SYSTEM}{random.choice(RAG_SYSTEM_PROMPTS)}{END}
{PROMPTER}Passage: {ctx}\n\nFrage: {question}{END}{BOT}{"irrelevant"}{END}"""

        all_rows.append(PROMPT)
        all_labels.append("classification")
        from_ds.append("deepset/germandpr")

    return all_rows, from_ds, all_labels
