import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def function_calling() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []

    ds = datasets.load_dataset("flozi00/german-function-calling", split="train")
    for row in tqdm(ds, desc="flozi00/german-function-calling"):
        try:
            prompt = row["Chat"]
            System = row["System"]
            prompt = prompt.replace("User:", PROMPTER)
            prompt = prompt.replace("ASSISTANT:", BOT)
            prompt = prompt.replace("<|endoftext|>", END)
            prompt = prompt.replace("USER:", PROMPTER)
            prompt = prompt.replace("Assistant:", BOT)
            System = System.replace("System:", SYSTEM)
            System = System.replace("SYSTEM:", SYSTEM)
            prompt = System + END + prompt
            all_rows.append(prompt)
            all_labels.append("function_calling")
            from_ds.append("flozi00/german-function-calling")
        except Exception as e:
            print(e)

    return all_rows, from_ds, all_labels
