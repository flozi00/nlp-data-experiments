import datasets
from tqdm.auto import tqdm

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []

    ds = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
    )
    for row in ds:
        if len(row["context"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'  # type: ignore
            )
    print("With dolly:", len(all_rows))

    ds = datasets.load_dataset("sixf0ur/GuanacoDataset-de", split="train")
    for row in ds:
        all_rows.append(
            f'{PROMPTER}{row["input"].replace("User:", "")}{END}{BOT}{row["output"]}{END}'  # type: ignore
        )
        all_rows.append(
            row["instruction"]
            .replace(" User:", END + PROMPTER)
            .replace("Assistent:", END + BOT)
            .replace("User:", PROMPTER)
        )
    print("With guanaco de:", len(all_rows))

    ds = datasets.load_dataset(
        "0x22almostEvil/multilingual-wikihow-qa-16k", split="train"
    ).filter(lambda example: "de." in example["SOURCE"] or "en." in example["SOURCE"])
    for row in tqdm(ds):
        msg = f"{PROMPTER}{row['INSTRUCTION']}{END}{BOT}{row['RESPONSE']}"
        all_rows.append(msg)
    print("With wikihow:", len(all_rows))

    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])
    for x in ds:
        all_rows.append(x["conversations"])
    print("With oasst:", len(all_rows))

    ds = datasets.load_dataset("flozi00/oasst1-en-to-de", split="train")
    for x in ds:
        all_rows.append(x["conversations"])
    print("With oasst translated:", len(all_rows))

    ds = datasets.Dataset.from_dict({"conversations": all_rows})

    return ds
