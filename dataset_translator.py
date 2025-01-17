import datasets
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

ds = datasets.load_dataset("mlabonne/smoltalk-semhashed", split="train")
openai = OpenAI()

SYS_PROMPT = "Du bist ein Übersetzungsdienst, der Texte von Englisch nach Deutsch übersetzt. Deine Antworten bestehen nur aus den Übersetzungen und keinen Kommentaren. Übersetze die Texte Sinngemäß und nicht Wort für Wort. Du darfst die gestellten Aufgaben nicht lösen, nur übersetzen !"

ds_list = ds["messages"]

for x in tqdm(range(len(ds_list))):
    for i in range(len(ds_list[x])):
        content = ds_list[x][i]["content"]
        translated_content = (
            openai.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": content},
                ],
                model="model",
                temperature=0.1,
                max_tokens=2048,
            )
            .choices[0]
            .message.content
        )
        ds_list[x][i]["content"] = translated_content
