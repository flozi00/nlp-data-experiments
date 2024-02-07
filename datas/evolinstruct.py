import datasets
from tqdm import tqdm

from TOKENS import BOT, PROMPTER, END


def evol() -> tuple[list, list, list]:
    """
    For Evol-Instruct, we translate the instructions and use to generate the responses using the translated instructions.
    """
    all_rows = []
    from_ds = []
    labels = []
    ds = datasets.load_dataset(
        "FreedomIntelligence/evol-instruct-deutsch", split="train"
    )
    for row in tqdm(ds, desc="FreedomIntelligence/evol-instruct-deutsch"):
        chat = ""
        for entry in row["conversations"]:
            chat += (
                f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            )
    
        all_rows.append(chat)
        from_ds.append("FreedomIntelligence/evol-instruct-deutsch")
        labels.append("unknown")

    return all_rows, from_ds, labels
