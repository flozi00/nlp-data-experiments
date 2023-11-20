import datasets
from TOKENS import BOT, PROMPTER, END
from tqdm import tqdm


def schnabeltier() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    ds = datasets.load_dataset("LeoLM/OpenSchnabeltier", split="train")
    for row in tqdm(ds, desc="LeoLM/OpenSchnabeltier"):
        prompt = f"{PROMPTER}{row['instruction_de']}{END}{BOT}{row['output_de']}{END}"
        all_rows.append(prompt)
        from_ds.append("LeoLM/OpenSchnabeltier")
        labels.append("unknown")

    return all_rows, from_ds, labels
