import datasets
from TOKENS import BOT, PROMPTER, END
from tqdm import tqdm


def german_poems() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    ds = datasets.load_dataset("LeoLM/German_Poems", split="train")
    for row in tqdm(ds, desc="LeoLM/German_Poems"):
        prompt = f"{PROMPTER}{row['prompt']}{END}{BOT}{row['poem']}{END}"
        all_rows.append(prompt)
        from_ds.append("LeoLM/German_Poems")
        labels.append("creative_writing")

    return all_rows, from_ds, labels
