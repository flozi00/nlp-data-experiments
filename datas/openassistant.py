import datasets
from TOKENS import BOT, PROMPTER, END
from tqdm import tqdm


def oa() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []
    ds = datasets.load_dataset(
        "blancsw/oasst2_top1_chat_format", split="train"
    ).filter(lambda x: x["langs"] == "de")
    for row in tqdm(ds, desc="blancsw/oasst2_top1_chat_format"):
        chat = ""
        for entry in row["conversation"]:
            chat += (
                f"{PROMPTER if entry['role'] == 'user' else BOT}{entry['content']}{END}"
            )
    
        all_rows.append(chat)
        from_ds.append("blancsw/oasst2_top1_chat_format")
        labels.append("unknown")

    return all_rows, from_ds, labels
