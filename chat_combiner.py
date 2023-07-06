import datasets

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []

    print("Argilla 15K")
    ds = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
    )
    for row in ds:
        if len(row["context"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'  # type: ignore
            )

    print("Guanaco German 17K")
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

    print("OASST filtered")
    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])
    for x in ds:
        all_rows.append(x["conversations"])

    print("german conversations")
    ds = datasets.load_dataset("flozi00/oasst1-en-to-de", split="train")
    for x in ds:
        all_rows.append(x["conversations"])

    ds = datasets.Dataset.from_dict({"conversations": all_rows})

    return ds
