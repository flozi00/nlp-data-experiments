import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm
import random

SYSTEM_PROMPTS = [
    """Im Folgenden beantwortet eine deutsche KI anhand der gegebenen passagen die Frage so gut wie möglich.
Bei der Beantwortung der Frage wird sich auf die passagen bezogen und keine Informationen ausgedacht.
Wenn die Beantwortung nicht möglich ist wird dies mitgeteilt.""",
]


def germanqa() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    SYSPrompt = random.choice(SYSTEM_PROMPTS)

    ds = datasets.load_dataset("flozi00/qa-tasks-german", split="train", streaming=True)
    for row in tqdm(ds, desc="flozi00/qa-tasks-german"):
        prompt = f"{SYSTEM}{SYSPrompt}{END}{PROMPTER}{row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(prompt)
        from_ds.append("flozi00/qa-tasks-german")
        labels.append("closed_qa")

    return all_rows, from_ds, labels
