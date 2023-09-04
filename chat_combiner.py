from collections import Counter
import datasets
import random
from rich.console import Console
from rich.table import Table

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def print_stats(stats):
    stats_keys = list(stats.keys())

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Column")
    table.add_column("Counts", justify="right")
    table.add_column("Percentage of dataset", justify="right")

    for k in stats_keys:
        table.add_row(
            str(k),
            str(stats[k]),
            str(stats[k] * percentage_multiplicator),
        )

    console.print(table)


def map_categories(cat):
    if cat in ["general_qa", "open_qa", "brainstorming", "classification"]:
        return "general"
    elif cat in ["closed_qa", "information_extraction", "summarization"]:
        return "information"
    elif cat in ["creative_writing"]:
        return "writing"


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    lang_id = []
    modes = []

    ds = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de"
    )
    for row in ds:
        all_rows.append(
            f'{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'
        )
        from_ds.append("argilla/databricks-dolly-15k-curated-multilingual")
        lang_id.append("de")
        modes.append(map_categories(row["category"]))

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
        modes.append("general")

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
            modes.append("general")

    ds = datasets.load_dataset("MBZUAI/Bactrian-X", "de", split="train")
    for row in ds:
        chat = f"{PROMPTER}{row['instruction']} {row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(chat)
        from_ds.append("MBZUAI/Bactrian-X")
        lang_id.append("de")
        modes.append("general")

    ds = datasets.load_dataset("deepset/germandpr", split="train")
    for row in ds:
        prompt = ""
        ctxs = []
        ctxs.extend(row["positive_ctxs"]["text"])
        ctxs.extend(row["negative_ctxs"]["text"])
        ctxs.extend(row["hard_negative_ctxs"]["text"])
        random.shuffle(ctxs)
        for ctx_id in range(len(ctxs)):
            ctx = ctxs[ctx_id]
            prompt += f"passage {ctx_id}: {ctx}\n"
        prompt += f"question: {row['question']}\n"
        prompt += f"answer: {row['answers'][0]}\n"

        all_rows.append(prompt)
        from_ds.append("deepset/germandpr")
        lang_id.append("de")
        modes.append(map_categories("closed_qa"))

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


final_data = get_chat_dataset()
percentage_multiplicator = 100 / len(final_data)
print_stats(Counter(final_data["from"]))
print_stats(Counter(final_data["mode"]))

final_data.push_to_hub("conversations")
