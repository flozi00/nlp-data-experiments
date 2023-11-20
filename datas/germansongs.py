import datasets
from TOKENS import BOT, PROMPTER, END
from tqdm import tqdm


def german_songs() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    ds = datasets.load_dataset("LeoLM/German_Songs", split="train")
    for row in tqdm(ds, desc="LeoLM/German_Songs"):
        prompt = f"{PROMPTER}{row['prompt']}{END}{BOT}{row['song']}{END}"
        all_rows.append(prompt)
        from_ds.append("LeoLM/German_Songs")
        labels.append("creative_writing")

        prompt = f"{PROMPTER}{row['song']}\n\n{row['analysis_prompt']}{END}{BOT}{row['analysis']}{END}"
        all_rows.append(prompt)
        from_ds.append("LeoLM/German_Songs")
        labels.append("creative_writing")

    return all_rows, from_ds, labels
