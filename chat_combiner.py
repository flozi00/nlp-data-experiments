from collections import Counter
import datasets

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"

dsets = [
    {
        "ds-name": "bjoernp/tagesschau-2018-2023",
        "text": "article",
    },
    {
        "ds-name": "SinclairSchneider/deutschlandfunk_de",
        "text": "content",
    },
    {
        "ds-name": "SinclairSchneider/bundeszentrale_fuer_politische_bildung",
        "text": "content",
    },
]


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    lang_id = []

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
            from_ds.append(fi)
            lang_id.append("de")

    ds = datasets.load_dataset("MBZUAI/Bactrian-X", "de", split="train")
    for row in ds:
        chat = f"{PROMPTER}{row['instruction']} {row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(chat)
        from_ds.append("MBZUAI/Bactrian-X")
        lang_id.append("de")

    for dset in dsets:
        ds_tg = datasets.load_dataset(dset["ds-name"], split="train")
        for ds in ds_tg:
            all_rows.append(ds[dset["text"]])
            from_ds.append(dset["ds-name"])
            lang_id.append("de")

    ds = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    ).filter(
        lambda example: example["lang"]
        in [
            "de",
        ]
    )
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

    print(Counter(ds["from"]))

    ds = ds.filter(lambda example: len(example["conversations"]) > 64 * 3)
    ds = ds.filter(lambda example: len(example["conversations"]) < 8192 * 3)

    print(Counter(ds["from"]))

    return ds
