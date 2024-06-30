import datasets


def math_dpo():
    prompt = []
    chosen = []
    rejected = []
    ds = datasets.load_dataset(
        "mayflowergmbh/distilabel-math-preference-dpo-de", split="train"
    )

    prompt = ds["input_translated"]
    chosen = ds["chosen_translated"]
    rejected = ds["rejected_translated"]

    return prompt, chosen, rejected
