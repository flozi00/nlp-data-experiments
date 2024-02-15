import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def self_instruct_gpt4() -> tuple[list, list, list]:
    all_rows = []
    all_labels = []
    from_ds = []

    ds = datasets.load_dataset("CausalLM/GPT-4-Self-Instruct-German", split="train")
    for row in tqdm(ds, desc="CausalLM/GPT-4-Self-Instruct-German"):
        try:
            prompt = PROMPTER + row["instruction"] + END + BOT + row["output"] + END
            all_rows.append(prompt)
            all_labels.append("unknown")
            from_ds.append("CausalLM/GPT-4-Self-Instruct-German")
        except Exception as e:
            print(e)

    return all_rows, from_ds, all_labels
