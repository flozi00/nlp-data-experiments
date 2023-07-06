import datasets

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

    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])

    for x in ds:
        all_rows.append(x["conversations"])

    ds = datasets.load_dataset("flozi00/german-conversations", split="train")
    for x in ds:
        all_rows.append(x["conversations"])

    ds = datasets.Dataset.from_dict({"conversations": all_rows})

    return ds
