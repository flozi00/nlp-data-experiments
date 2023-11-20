import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm
import random

SYSTEM_PROMPTS = [
    "Im folgenden werden aus Konversationen eigenständige und ausformulierte Sätze gebildet, wenn dies notwendig ist damit alle Informationen enthalten sind.",
    "Ein System formuliert eine Konversation in eigenständige und ausformulierte Sätze um.",
    "Im folgenden wird eine Konversation in eigenständige und ausformulierte Sätze umgeformt.",
]


def single_queries() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    SYSPrompt = random.choice(SYSTEM_PROMPTS)

    ds = datasets.load_dataset(
        "flozi00/single-queries-german", split="train", streaming=True
    )
    for row in tqdm(ds, desc="flozi00/single-queries-german"):
        prompt = f"{SYSTEM}{SYSPrompt}{END}{PROMPTER}{row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(prompt)
        from_ds.append("flozi00/single-queries-german")
        labels.append("unknown")

    return all_rows, from_ds, labels
