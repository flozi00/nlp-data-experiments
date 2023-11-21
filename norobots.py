import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM

ds = datasets.load_dataset("HuggingFaceH4/no_robots", split="train_sft")
ds2 = datasets.load_dataset("flozi00/no_robots_german", split="train")

texts = ds2["messages"]
labels = ds["category"][: len(texts)]

print(len(texts), len(labels))

ds = datasets.Dataset.from_dict({"messages": texts, "category": labels})

ds.push_to_hub("no_robots_german")
