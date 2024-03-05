import datasets
from difflib import SequenceMatcher

ds = datasets.load_dataset("DIBT/10k_prompts_ranked", split="train")

ds = ds.filter(lambda x: len(x["prompt"]) >= 5)
ds = ds.filter(lambda x: len(x["prompt"]) <= 1024)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() < 0.8


def rem_lines(batch):
    batch["prompt"] = batch["prompt"].replace("\n", " /n ").replace("\r", " /r ")
    return batch


ds = ds.map(rem_lines)

print(ds)

with open("dibt.txt", "w", encoding="utf-8") as f:
    for example in ds:
        f.write(example["prompt"] + "\n")


with open("dibt_de.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

ds = ds.add_column("prompt_de", lines)

print(ds)
ds = ds.filter(lambda x: similar(x["prompt_de"], x["prompt"]))
print(ds)


def add_lines(batch):
    batch["prompt_de"] = batch["prompt_de"].replace(" /n ", "\n").replace(" /r ", "\r")
    batch["prompt"] = batch["prompt"].replace(" /n ", "\n").replace(" /r ", "\r")

    return batch


ds = ds.map(add_lines)

ds.push_to_hub("dibt_de")
