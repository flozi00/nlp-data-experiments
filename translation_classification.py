import datasets
import json
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient


client = InferenceClient(model="http://localhost:1337")


system_content = """You are an classiyfier for translation quality. Your task is to classify the quality of the translation into one of the following categories: Good, Bad, or Neutral.

Good: The translation is accurate and fluent. Every word is translated correctly and the grammar is perfect. No mistakes are present in the translation, not even an charakter. The translation pair is perfect.
Neutral: Every word is translated correctly and the grammar is perfect. But the translation is not fluent. The translation is not perfect, but it's not bad either.
Bad: The translation is not accurate and/or not fluent. The translation is not perfect and the grammar is not perfect. The translation pair is bad. There is at least one mistake in the translation.

Examples:

de: Ich bin ein Student. -> en: I am a student. -> Good
de: Legen Sie die Endzeit fest.@option:check -> en: Set the end time -> Bad
de: Ich liebe es bei sonnigem Wetter spazieren zu gehen. -> en: I love going for walks at sunny weather -> Neutral
de: Ich bin ein Student. -> en: I am an student. -> Neutral
de: Also, 90 Prozent meiner fotografischen Arbeit ist genau genommen gar nicht fotografisch -> en: Okay, so 90 percent of my photographic process is, in fact, not photographic. -> Good
de: Bis bald, Simon. -> en: you too. -> Bad
de: Nehmen sie Platz. -> en: Please sit. -> Neutral
de: Vega -> en: - Vega -> Neutral
de: An manchen Tagen fühle ich mich nicht so gut. -> en: On some days I doesn't feel so good. -> Neutral
de: Deine Habgier wird noch dein Tod sein. -> en: It's greed that it's gonna be the death of you, 'cause you... -> Bad
de: Sagen Sie einfach stopp. -> en: Just say when. -> Bad
de: Ich darf wirklich aufstehen, Herr Hofrat? -> en: I am really allowed to get up, Doctor? -> Bad
de: Nehmen sie Platz. -> en: Please take a seat. -> Good

"""


try:
    with open("translation_classification.json", "r", encoding="utf-8") as f:
        datas = json.loads(f.read())
except FileNotFoundError:
    datas = []

ds = datasets.load_dataset("Helsinki-NLP/opus-100", "de-en", split="train").filter(
    lambda x: len(x["translation"]["de"]) >= 10 and len(x["translation"]["en"]) <= 1024
)

COUNTING = 0
for example in tqdm(ds, desc="Helsinki-NLP/opus-100"):
    de = example["translation"]["de"]
    en = example["translation"]["en"]
    prompt = f"de: {de} -> en: {en}"

    if any(d["de"] == de and d["en"] == en for d in datas):
        continue

    response = client.text_generation(
        prompt=f"{system_content + prompt} ->",
        max_new_tokens=4,
        temperature=0.01,
        do_sample=False,
    )

    label = (
        "Neutral"
        if "Neutral" in response
        else ("Good" if "Good" in response else "Bad" if "Bad" in response else "None")
    )
    if label != "None":
        datas.append(
            {"de": de, "en": en, "label": label, "text": f"en: {en} --> de: {de}"}
        )

    if COUNTING % 100 == 0:
        with open("translation_classification.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(datas, indent=4, ensure_ascii=False))

    COUNTING += 1
