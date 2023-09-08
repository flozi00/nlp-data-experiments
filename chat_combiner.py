from collections import Counter
from typing import Literal
import datasets
import random
from rich.console import Console
from rich.table import Table
from langdetect import detect
from tqdm import tqdm
from system_prompts import *
from TOKENS import *
import torch
from transformers import pipeline
from optimum.bettertransformer import BetterTransformer
from filecache import filecache



pipe = pipeline("text2text-generation", model="flozi00/t5-small-llm-tasks", device=0, torch_dtype=torch.float16)
pipe.model = BetterTransformer.transform(pipe.model)

@filecache(24 * 60 * 60)
def get_dolly_label(prompt: str) -> str:
    return pipe(f"{PROMPTER}{prompt}{END}", max_new_tokens = 5, do_sample=False)[0]["generated_text"]

def print_stats(stats) -> None:
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


def map_categories(cat) -> Literal['general', 'information', 'writing'] | None:
    if cat in ["general_qa", "open_qa", "brainstorming"]:
        return "general"
    elif cat in ["closed_qa", "information_extraction", "summarization", "classification"]:
        return "information"
    elif cat in ["creative_writing", "de-summarize"]:
        return "writing"


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    lang_id = []
    modes = []

    ds = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de"
    )
    for row in tqdm(ds, desc="Databricks Dolly"):
        all_rows.append(
            f'{SYSTEM}{random.choice(general_system_prompts)}{END}{PROMPTER}{row["context"]}\n{row["instruction"]}{END}{BOT}{row["response"]}{END}'
        )
        from_ds.append("argilla/databricks-dolly-15k-curated-multilingual")
        lang_id.append("de")
        modes.append(map_categories(row["category"]))

    ds = datasets.load_dataset("FreedomIntelligence/evol-instruct-deutsch", split="train")
    for row in tqdm(ds, desc="FreedomIntelligence/evol-instruct-deutsch"):
        chat = ""
        label = row["conversations"][0]["value"]
        label = get_dolly_label(label)
        label = map_categories(label)
        for entry in row["conversations"]:
            chat += f"{SYSTEM}{random.choice(general_system_prompts)}{END}{PROMPTER if entry['from'] == 'human' else BOT}{entry['value']}{END}"
        all_rows.append(chat)
        from_ds.append("FreedomIntelligence/evol-instruct-deutsch")
        lang_id.append("de")
        modes.append(label)

    ds = datasets.load_dataset("MBZUAI/Bactrian-X", "de", split="train")
    
    for row in tqdm(ds, desc="Bactrian-X"):
        label = get_dolly_label(row["instruction"])
        label = map_categories(label)
        chat = f"{SYSTEM}{random.choice(general_system_prompts)}{END}{PROMPTER}{row['instruction']} {row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(chat)
        from_ds.append("MBZUAI/Bactrian-X")
        lang_id.append("de")
        modes.append(label)

    ds = datasets.load_dataset("deepset/germandpr", split="train")
    for row in tqdm(ds, desc="German DPR"):
        prompt = f"{SYSTEM}{random.choice(short_qa_system_prompts)}{END}"
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

    ds = datasets.load_dataset("snipaid/instruct-snippet-mlsum-v2", split="train")
    for row in tqdm(ds, desc="Instruct Snippet MLSum"):
        prompt = f"{SYSTEM}{random.choice(general_system_prompts)}{END}{PROMPTER}{row['instruction']}\n{row['input']}{END}{BOT}{row['output']}{END}"
        all_rows.append(prompt)
        from_ds.append("snipaid/instruct-snippet-mlsum-v2")
        lang_id.append("de")
        modes.append(map_categories("summarization"))

    ds = datasets.load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
    for row in tqdm(ds, desc="OpenAssistant"):
        try:
            prompt = row["text"]
            prompt = prompt.replace("<|im_start|>user", PROMPTER)
            prompt = prompt.replace("<|im_start|>assistant", BOT)
            prompt = prompt.replace("<|im_end|>", END)
            lang = detect(prompt)
            if lang != "de":
                continue
            prompt = f"{SYSTEM}{random.choice(oa_system_prompts)}{END}{prompt}"
            all_rows.append(prompt)
            from_ds.append("OpenAssistant/oasst_top1_2023-08-25")
            lang_id.append(lang)
            modes.append("general")
        except Exception as e:
            print(e)

    ds = datasets.Dataset.from_dict(
        {
            "conversations": all_rows,
            "from": from_ds,
            "lang": lang_id,
            "mode": modes,
            "chars": [len(x) for x in all_rows],
        }
    )

    ds = ds.filter(lambda example: example["chars"] > 64 * 3)
    
    return ds


final_data = get_chat_dataset()
percentage_multiplicator = 100 / len(final_data)
print_stats(Counter(final_data["from"]))
print_stats(Counter(final_data["mode"]))
print_stats(Counter(final_data["lang"]))

final_data.push_to_hub("conversations", max_shard_size="1GB")
