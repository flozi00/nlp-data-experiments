import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def ultra_chat() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []

    ds = datasets.load_dataset("mayflowergmbh/ultra-chat_de", split="train")
    for row in tqdm(ds, desc="mayflowergmbh/ultra-chat_de"):
        try:
            prompt = SYSTEM + row["instruction"] + END + PROMPTER + row["input"] + END + BOT + row["output"] + END
            all_rows.append(prompt)
            all_labels.append("unknown")
            from_ds.append("mayflowergmbh/ultra-chat_de")
        except Exception as e:
            print(e)

    return all_rows, from_ds, all_labels
