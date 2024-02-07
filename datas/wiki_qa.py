import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def wiki_qa() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []

    ds = datasets.load_dataset("mayflowergmbh/wiki_qa_de", split="train")
    for row in tqdm(ds, desc="mayflowergmbh/wiki_qa_de"):
        try:
            prompt = SYSTEM + row["instruction"] + END + PROMPTER + row["input"] + END + BOT + row["output"] + END
            all_rows.append(prompt)
            all_labels.append("closed_qa")
            from_ds.append("mayflowergmbh/wiki_qa_de")
        except Exception as e:
            print(e)

    return all_rows, from_ds, all_labels
