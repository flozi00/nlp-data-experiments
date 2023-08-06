import datasets

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    lang_id = []

    print("argilla/databricks-dolly-15k-curated-multilingual")
    for lang in ["de"]:
        # Dolly multilingual
        ds = datasets.load_dataset(
            "argilla/databricks-dolly-15k-curated-multilingual", split=lang
        )
        for row in ds:
            if len(row["context"]) > 128:  # type: ignore
                all_rows.append(
                    f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'  # type: ignore
                )
                from_ds.append("argilla/databricks-dolly-15k-curated-multilingual")
                lang_id.append(lang)

    # Guanaco
    print("sixf0ur/GuanacoDataset-de")
    ds = datasets.load_dataset("sixf0ur/GuanacoDataset-de", split="train")
    for row in ds:
        all_rows.append(
            f'{PROMPTER}{row["input"].replace("User:", "")}{END}{BOT}{row["output"]}{END}'  # type: ignore
        )
        from_ds.append("sixf0ur/GuanacoDataset-de")
        lang_id.append("de")
        all_rows.append(
            row["instruction"]
            .replace(" User:", END + PROMPTER)
            .replace("Assistent:", END + BOT)
            .replace("User:", PROMPTER)
        )
        from_ds.append("sixf0ur/GuanacoDataset-de")
        lang_id.append("de")

    # wikihow, german and english
    print("0x22almostEvil/multilingual-wikihow-qa-16k")
    ds = datasets.load_dataset(
        "0x22almostEvil/multilingual-wikihow-qa-16k", split="train"
    ).filter(lambda example: "de." in example["SOURCE"] or "en." in example["SOURCE"])
    for row in ds:
        msg = f"{PROMPTER}{row['INSTRUCTION']}{END}{BOT}{row['RESPONSE']}{END}"
        all_rows.append(msg)
        from_ds.append("0x22almostEvil/multilingual-wikihow-qa-16k")
        lang_id.append("en" if "en." in row["SOURCE"] else "de")

    # wizard vicuna german
    print("musabg/wizard_vicuna_70k_unfiltered_de")
    ds = datasets.load_dataset("musabg/wizard_vicuna_70k_unfiltered_de", split="train")
    for row in ds:
        chat = ""
        for entry in row["conversations"]:
            chat += (
                f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            )
        all_rows.append(chat)
        from_ds.append("musabg/wizard_vicuna_70k_unfiltered_de")
        lang_id.append("de")

    for fi in [
        "FreedomIntelligence/alpaca-gpt4-deutsch",
        "FreedomIntelligence/evol-instruct-deutsch",
    ]:
        ds = datasets.load_dataset(fi, split="train")
        for row in ds:
            chat = ""
            for entry in row["conversations"]:
                chat += f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            all_rows.append(chat)
            from_ds.append("FreedomIntelligence/evol-instruct-deutsch")
            lang_id.append("de")

    # openorca
    print("Open-Orca/OpenOrca")
    ds = datasets.load_dataset("Open-Orca/OpenOrca", split="train").filter(
        lambda example: "cot." in example["id"]
    )
    for row in ds:
        msg = f"{PROMPTER}{row['question']}{END}{BOT}{row['response']}{END}"
        all_rows.append(msg)
        from_ds.append("Open-Orca/OpenOrca")
        lang_id.append("en")

    # oasst
    print("OpenAssistant Datasets")
    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(lambda example: example["lang"] in ["de", "en"])
    for x in ds:
        all_rows.append(x["conversations"])
        from_ds.append("flozi00/openassistant-oasst1-flattened-filtered")
        lang_id.append(x["lang"])

    # oasst translated
    ds = datasets.load_dataset("flozi00/oasst1-en-to-de", split="train")
    for x in ds:
        all_rows.append(x["conversations"])
        from_ds.append("flozi00/oasst1-en-to-de")
        lang_id.append("de")

    ds = datasets.Dataset.from_dict(
        {"conversations": all_rows, "from": from_ds, "lang": lang_id}
    )

    ds = ds.filter(lambda example: len(example["conversations"]) < 7168 * 3)
    ds = ds.filter(lambda example: len(example["conversations"]) > 256 * 3)

    return ds
