import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def no_robots() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    ds = datasets.load_dataset("flozi00/no_robots_german", split="train")
    for row in tqdm(ds, desc="no_robots_german"):
        try:
            prompt = row["messages"]
            prompt = prompt.replace("### User:", PROMPTER)
            prompt = prompt.replace("### Assistant:", BOT)
            prompt = prompt.replace("### System:", SYSTEM)
            prompt = prompt.replace("</s>", END)
            all_rows.append(prompt)
            from_ds.append("flozi00/no_robots_german")
            labels.append("unknown")
        except Exception as e:
            print(e)

    return all_rows, from_ds, labels
