import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
from tqdm import tqdm


def get_label(label):
    if label in ["Generation", "Rewrite"]:
        return "creative_writing"

    if label in ["Open QA"]:
        return "open_qa"

    if label in ["Brainstorm"]:
        return "brainstorming"

    if label in ["Summarize"]:
        return "summarization"

    if label in ["Classify"]:
        return "classification"

    if label in ["Closed QA"]:
        return "closed_qa"

    if label in ["Extract"]:
        return "information_extraction"

    if label in ["Coding"]:
        return "coding"

    return "unknown"


def no_robots() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []

    ds = datasets.load_dataset("flozi00/no_robots_german", split="train")
    for row in tqdm(ds, desc="no_robots_german"):
        try:
            converted_label = get_label(row["category"])
            prompt = row["messages"]
            prompt = prompt.replace("### User:", PROMPTER)
            prompt = prompt.replace("### Assistant:", END + BOT)
            prompt = prompt.replace("### System:", SYSTEM)
            prompt = prompt.replace("</s>", END)
            prompt = prompt.replace(END + END, END)
            all_rows.append(prompt)
            from_ds.append("flozi00/no_robots_german")
            labels.append(converted_label)
        except Exception as e:
            print(e)

    return all_rows, from_ds, labels
