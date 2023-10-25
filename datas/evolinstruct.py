import datasets
from tqdm import tqdm

from TOKENS import BOT, PROMPTER, END
from utils.detector import detector
from utils.classifier import get_dolly_label


def evol() -> tuple[list, list, list]:
    """
    For Evol-Instruct, we translate the instructions and use to generate the responses using the translated instructions.
    """
    all_rows = []
    all_labels = []
    from_ds = []
    ds = datasets.load_dataset(
        "FreedomIntelligence/evol-instruct-deutsch", split="train"
    )
    for row in tqdm(ds, desc="FreedomIntelligence/evol-instruct-deutsch"):
        chat = ""
        for entry in row["conversations"]:
            chat += (
                f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            )
        if (
            detector(row["conversations"][0]["value"])
            == detector(row["conversations"][1]["value"])
            == "de"
        ):
            all_rows.append(chat)
            all_labels.append(get_dolly_label(row["conversations"][0]["value"]))
            from_ds.append("FreedomIntelligence/evol-instruct-deutsch")

    return all_rows, all_labels, from_ds
