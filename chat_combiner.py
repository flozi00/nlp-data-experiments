from collections import Counter
import datasets

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"
SEQ_LENGTH = 4096


def combine_strings(strings):
    result = []
    current_string = strings[0]
    for string in strings[1:]:
        if len(current_string + string) <= SEQ_LENGTH * 3:
            current_string += string
        else:
            result.append(current_string)
            current_string = string
    result.append(current_string)
    return result


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    lang_id = []
    modes = []

    for lang in ["de", "en"]:
        # Dolly multilingual
        ds = datasets.load_dataset(
            "argilla/databricks-dolly-15k-curated-multilingual", split=lang
        )
        temp_list = []
        for row in ds:
            temp_list.append(
                f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'
            )
        temp_list = combine_strings(temp_list)
        for row in temp_list:
            all_rows.append(row)
            from_ds.append("argilla/databricks-dolly-15k-curated-multilingual")
            lang_id.append(lang)
            modes.append("fine-tune")

    ds = datasets.load_dataset("sixf0ur/GuanacoDataset-de", split="train")
    temp_list = []
    for row in ds:
        temp_list.append(
            f'{PROMPTER}{row["input"].replace("User:", "")}{END}{BOT}{row["output"]}{END}'
        )
        temp_list.append(
            row["instruction"]
            .replace(" User:", END + PROMPTER)
            .replace("Assistent:", END + BOT)
            .replace("User:", PROMPTER)
        )
    temp_list = combine_strings(temp_list)
    for row in temp_list:
        all_rows.append(row)
        from_ds.append("sixf0ur/GuanacoDataset-de")
        lang_id.append("de")
        modes.append("fine-tune")

    ds = datasets.load_dataset("musabg/wizard_vicuna_70k_unfiltered_de", split="train")
    temp_list = []
    for row in ds:
        chat = ""
        for entry in row["conversations"]:
            chat += (
                f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            )
        temp_list.append(chat)
    temp_list = combine_strings(temp_list)
    for row in temp_list:
        all_rows.append(row)
        from_ds.append("musabg/wizard_vicuna_70k_unfiltered_de")
        lang_id.append("de")
        modes.append("fine-tune")

    for fi in [
        "FreedomIntelligence/alpaca-gpt4-deutsch",
        "FreedomIntelligence/evol-instruct-deutsch",
    ]:
        ds = datasets.load_dataset(fi, split="train")
        temp_list = []
        for row in ds:
            chat = ""
            for entry in row["conversations"]:
                chat += f"{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
            temp_list.append(chat)
        temp_list = combine_strings(temp_list)
        for row in temp_list:
            all_rows.append(row)
            from_ds.append(fi)
            lang_id.append("de")
            modes.append("fine-tune")

    for lang in ["de", "en"]:
        ds = datasets.load_dataset("MBZUAI/Bactrian-X", lang, split="train")
        temp_list = []
        for row in ds:
            chat = f"{PROMPTER}{row['instruction']} {row['input']}{END}{BOT}{row['output']}{END}"
            temp_list.append(chat)
        temp_list = combine_strings(temp_list)
        for row in temp_list:
            all_rows.append(row)
            from_ds.append("MBZUAI/Bactrian-X")
            lang_id.append("de")
            modes.append("fine-tune")

    for lang in ["de", "en"]:
        ds = datasets.load_dataset(
            "flozi00/openassistant-oasst1-flattened-filtered", split="train"
        ).filter(lambda example: example["lang"] == lang)
        temp_list = []
        for x in ds:
            temp_list.append(x["conversations"])
        temp_list = combine_strings(temp_list)
        for row in temp_list:
            all_rows.append(row)
            from_ds.append("flozi00/openassistant-oasst1-flattened-filtered")
            lang_id.append(lang)
            modes.append("fine-tune")

    # oasst translated
    ds = datasets.load_dataset("flozi00/oasst1-en-to-de", split="train")
    temp_list = []
    for x in ds:
        temp_list.append(x["conversations"])
    temp_list = combine_strings(temp_list)
    for row in temp_list:
        all_rows.append(row)
        from_ds.append("flozi00/oasst1-en-to-de")
        lang_id.append("de")
        modes.append("fine-tune")

    ds = datasets.Dataset.from_dict(
        {
            "conversations": all_rows,
            "from": from_ds,
            "lang": lang_id,
            "mode": modes,
            "chars": [len(x) for x in all_rows],
        }
    )

    print(Counter(ds["from"]))

    ds = ds.filter(lambda example: example["chars"] > 64 * 3)
    ds = ds.filter(lambda example: example["chars"] < 8192 * 3)

    print(Counter(ds["from"]))

    return ds
