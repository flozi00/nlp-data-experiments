import datasets
import random

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"

WRITER_PREFIXES = [
    "Schreibe einen Artikel über folgendes Thema: ",
    "Schreibe ein Text zu folgendem Thema: ",
    "Ich möchte einen Text zu folgendem Thema: ",
    "Verfasse einen Blogpost: ",
    "Erstelle einen Artikel: ",
    "Erstelle einen Text: ",
    "Erstelle einen Blogpost: ",
    "Erstelle einen Artikel: ",
    "Denk dir einen Artikel aus: ",
]


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []

    ds = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
    )
    for row in ds:
        if len(row["context"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'  # type: ignore
            )
            from_ds.append("argilla/databricks-dolly-15k-curated-multilingual")
    print("With dolly:", len(all_rows))

    ds = datasets.load_dataset("sixf0ur/GuanacoDataset-de", split="train")
    for row in ds:
        all_rows.append(
            f'{PROMPTER}{row["input"].replace("User:", "")}{END}{BOT}{row["output"]}{END}'  # type: ignore
        )
        from_ds.append("sixf0ur/GuanacoDataset-de")
        all_rows.append(
            row["instruction"]
            .replace(" User:", END + PROMPTER)
            .replace("Assistent:", END + BOT)
            .replace("User:", PROMPTER)
        )
        from_ds.append("sixf0ur/GuanacoDataset-de")
    print("With guanaco de:", len(all_rows))

    ds = datasets.load_dataset(
        "0x22almostEvil/multilingual-wikihow-qa-16k", split="train"
    ).filter(lambda example: "de." in example["SOURCE"] or "en." in example["SOURCE"])
    for row in ds:
        msg = f"{PROMPTER}{row['INSTRUCTION']}{END}{BOT}{row['RESPONSE']}{END}"
        all_rows.append(msg)
        from_ds.append("0x22almostEvil/multilingual-wikihow-qa-16k")
    print("With wikihow:", len(all_rows))

    ds = datasets.load_dataset("musabg/wizard_vicuna_70k_unfiltered_de", split="train")
    for row in ds:
        chat = ""
        for entry in row["conversations"]:
            chat += (
                f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            )
        all_rows.append(chat)
        from_ds.append("musabg/wizard_vicuna_70k_unfiltered_de")
    print("With wizard vicuna:", len(all_rows))

    ds = datasets.load_dataset("mlsum", "de", split="train").filter(
        lambda example, idx: idx % 10 == 0, with_indices=True
    )
    for row in ds:
        text = (
            PROMPTER
            + random.choice(WRITER_PREFIXES)
            + row[random.choice(["title", "summary"])]
            + f"{END}{BOT}"
        )
        all_rows.append(text)
        from_ds.append("mlsum")
    print("With mlsum:", len(all_rows))

    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])
    for x in ds:
        all_rows.append(x["conversations"])
        from_ds.append("flozi00/openassistant-oasst1-flattened-filtered")
    print("With oasst:", len(all_rows))

    ds = datasets.load_dataset("flozi00/oasst1-en-to-de", split="train")
    for x in ds:
        all_rows.append(x["conversations"])
        from_ds.append("flozi00/oasst1-en-to-de")
    print("With oasst translated:", len(all_rows))

    ds = datasets.Dataset.from_dict({"conversations": all_rows, "from": from_ds})

    return ds
