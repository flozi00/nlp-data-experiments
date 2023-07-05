with open("_oasst_en_0.txt", "r") as myfile:
    chats = myfile.read()

chats = chats.split("\n******\n")

print(len(chats))

for x in range(len(chats)):
    chat = chats[x]
    chat = chat.split("\n\n")
    chat_str = ""
    for y in range(len(chat)):
        if len(chat[y].strip()) > 3:
            if y == 0:
                chat_str += f"<|prompter|>{chat[y].strip()}<|endoftext|>"
            elif y % 2 != 0:
                chat_str += f"<|assistant|>{chat[y].strip()}<|endoftext|>"
            elif y % 2 == 0:
                chat_str += f"<|prompter|>{chat[y].strip()}<|endoftext|>"
    chats[x] = chat_str

import datasets

ds = datasets.Dataset.from_dict({"conversations": chats})

ds.push_to_hub("german-conversations")
