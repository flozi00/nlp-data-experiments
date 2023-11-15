import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM

ds = datasets.load_dataset("HuggingFaceH4/no_robots", split="train_sft")

entries = []

for entry in ds["messages"]:
    text = ""
    for message in entry:
        if message["role"] == "user":
            tok = PROMPTER
        elif message["role"] == "assistant":
            tok = BOT
        else:
            tok = SYSTEM

        text += f"{tok}{message['content']}{END}\n"
    
    entries.append(text)

with open("no_robots.txt", "w") as f:
    for entry in entries:
        f.write(entry)
        f.write("\n***\n")