import openai
import datasets
import json
import huggingface_hub
from tqdm.auto import tqdm

client = openai.OpenAI(
    api_key=huggingface_hub.get_token(),
    base_url="http://localhost:1337/v1/",
)

system_content = """You are an classiyfier for translation quality. Your task is to classify the quality of the translation into one of the following categories: Good, Bad, or Neutral.

Good: The translation is accurate and fluent. Every word is translated correctly and the grammar is perfect. No mistakes are present in the translation, not even an charakter.
Neutral: There are some small mistakes in the translation like single charakter mistakes or single words could be better.
Bad: The translation is not accurate or the grammar is incorrect. More than one mistake is present in the translation.

Examples:

de: Ich bin ein Student. -> en: I am a student. -> Good
de: Legen Sie die Endzeit fest.@option:check -> en: Set the end time -> Bad
de: Ich liebe es bei sonnigem Wetter spazieren zu gehen. -> en: I love to go at sunny weather. -> Neutral
de: Ich bin ein Student. -> en: I am an student. -> Neutral
de: Bis bald, Simon. -> en: you too. -> Bad
de: Vega -> en: - Vega -> Neutral
de: An manchen Tagen fühle ich mich nicht so gut. -> en: On some days I doesn't feel so good. -> Neutral
de: Deine Habgier wird noch dein Tod sein. -> en: It's greed that it's gonna be the death of you, 'cause you... -> Bad
de: Sagen Sie einfach stopp. -> en: Just say when. -> Bad

Answer the classification of the translation pair only ! Do not explain your answer or correct the translation.
"""


try:
    with open("translation_classification.json", "r", encoding="utf-8") as f:
        datas = json.loads(f.read())
except FileNotFoundError:
    datas = []

ds = datasets.load_dataset("Helsinki-NLP/opus-100", "de-en", split="train").filter(
    lambda x: len(x["translation"]["de"]) >= 5
)

ds_alma = datasets.load_dataset("haoranxu/ALMA-Human-Parallel", "de-en", split="train")
for example in tqdm(ds_alma, desc="ALMA-Human-Parallel"):
    de = example["translation"]["de"]
    en = example["translation"]["en"]
    if any(d["de"] == de and d["en"] == en for d in datas):
        continue
    datas.append({"de": de, "en": en, "label": "Good"})

ds_dibt = datasets.load_dataset("flozi00/dibt_de", split="train")
for example in tqdm(ds_dibt, desc="dibt_de"):
    de = example["prompt_de"]
    en = example["prompt"]
    if any(d["de"] == de and d["en"] == en for d in datas):
        continue
    datas.append({"de": de, "en": en, "label": "Good"})

ds_dolly = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de"
)
for example in tqdm(ds_dolly, desc="databricks-dolly-15k-curated-multilingual"):
    de = example["instruction"]
    en = example["instruction_original_en"]
    if any(d["de"] == de and d["en"] == en for d in datas):
        continue
    datas.append({"de": de, "en": en, "label": "Good"})

    de = example["response"]
    en = example["response_original_en"]
    if any(d["de"] == de and d["en"] == en for d in datas):
        continue
    datas.append({"de": de, "en": en, "label": "Good"})

    de = example["context"]
    en = example["context_original_en"]
    if any(d["de"] == de and d["en"] == en for d in datas):
        continue
    if len(de) < 5 or len(en) < 5:
        continue
    datas.append({"de": de, "en": en, "label": "Good"})

COUNTING = 0
for example in tqdm(ds, desc="Helsinki-NLP/opus-100"):
    de = example["translation"]["de"]
    en = example["translation"]["en"]
    prompt = f"de: {de} -> en: {en}"

    if any(d["de"] == de and d["en"] == en for d in datas):
        continue

    if len(de.split(" ")) > len(en.split(" ") * 2) or len(en.split(" ")) > len(
        de.split(" ") * 2
    ):
        label = "Bad"
    else:
        chat_completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "user", "content": system_content},
                {
                    "role": "assistant",
                    "content": "Okay, let's start with the first example.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4,
        )
        response = chat_completion.choices[0].message.content

        label = (
            "Neutral"
            if "Neutral" in response
            else (
                "Good" if "Good" in response else "Bad" if "Bad" in response else "None"
            )
        )

    datas.append({"de": de, "en": en, "label": label, "text": f"en: {en} --> de: {de}"})

    if COUNTING % 100 == 0:
        # for i in range(len(datas)):
        #    datas[i]["text"] = f"en: {datas[i]['en']} --> de: {datas[i]['de']}"
        with open("translation_classification.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(datas, indent=4, ensure_ascii=False))

    COUNTING += 1
