import datasets
from TOKENS import BOT, PROMPTER, END
from utils.detector import detector
from tqdm import tqdm


def oa() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []

    ds = datasets.load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
    for row in tqdm(ds, desc="OpenAssistant"):
        try:
            prompt = row["text"]
            prompt = prompt.replace("<|im_start|>user", PROMPTER)
            prompt = prompt.replace("<|im_start|>assistant", BOT)
            prompt = prompt.replace("<|im_end|>", END)
            if detector(prompt) != "de":
                continue
            all_rows.append(prompt)
            all_labels.append("chat")
            from_ds.append("OpenAssistant/oasst_top1_2023-08-25")
        except Exception as e:
            print(e)

    return all_rows, all_labels, from_ds
