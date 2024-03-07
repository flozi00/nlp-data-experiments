import openai
import huggingface_hub
import datasets

client = openai.OpenAI(
    api_key=huggingface_hub.get_token(),
    base_url="http://localhost:1337/v1/",
)

system_content = """You are an classiyfier for translation quality. Your task is to classify the quality of the translation into one of the following categories: Perfect, Bad, or Good.

Perfect: The translation is accurate and fluent. Every word is translated correctly and the grammar is perfect. No mistakes are present in the translation, not even an charakter. The translation pair is perfect.
Good: Every word is translated correctly and the grammar is perfect. But the translation is not fluent. The translation is not perfect, but it's not bad either.
Bad: The translation is not accurate and/or not fluent. The translation is not perfect and the grammar is not perfect. The translation pair is bad. There is at least one mistake in the translation.

Examples:

de: Ich bin ein Student. -> en: I am a student. -> Perfect
de: Legen Sie die Endzeit fest.@option:check -> en: Set the end time -> Bad
de: Ich liebe es bei sonnigem Wetter spazieren zu gehen. -> en: I love going for walks at sunny weather -> Good
de: Ich bin ein Student. -> en: I am an student. -> Good
de: Also, 90 Prozent meiner fotografischen Arbeit ist genau genommen gar nicht fotografisch -> en: Okay, so 90 percent of my photographic process is, in fact, not photographic. -> Perfect
de: Bis bald, Simon. -> en: you too. -> Bad
de: Nehmen sie Platz. -> en: Please sit. -> Good
de: Vega -> en: - Vega -> Good
de: An manchen Tagen fühle ich mich nicht so gut. -> en: On some days I doesn't feel so Perfect. -> Good
de: Deine Habgier wird noch dein Tod sein. -> en: It's greed that it's gonna be the death of you, 'cause you... -> Bad
de: Sagen Sie einfach stopp. -> en: Just say when. -> Bad
de: Ich darf wirklich aufstehen, Herr Hofrat? -> en: I am really allowed to get up, Doctor? -> Bad
de: Nehmen sie Platz. -> en: Please take a seat. -> Perfect
de: in Erwägung nachstehenden Grundes: -> en: whereas: -> Bad
de: - Danke. - Okay. -> en: Thank you. -> Bad
de: Eins, zwei, eins, zwei. -> en: One, two, one, two. -> Perfect
de: Eins, zwei, eins, zwei. -> en: slow, slow, quick, quick -> Bad

Answer the classification of the translation pair only ! Do not explain your answer or correct the translation.
"""


def rate_translation(example):
    de: str = example["translation"]["de"]
    en: str = example["translation"]["en"]
    prompt = f"de: {de} -> en: {en}"

    if len(de.split(" ")) > len(en.split(" ") * 2) or len(en.split(" ")) > len(
        de.split(" ") * 2
    ):
        label = "Bad"
    else:
        chat_completion = client.chat.completions.create(
            model="",
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
            "Good"
            if "Good" in response
            else ("Perfect" if "Perfect" in response else "Bad")
        )

    example["label"] = label

    return example


DS_NAME = "haoranxu/ALMA-Human-Parallel"

if __name__ == "__main__":
    ds = datasets.load_dataset(DS_NAME, "de-en", split="train").filter(
        lambda x: len(x["translation"]["de"]) >= 5
        and len(x["translation"]["en"]) <= 2048
    )
    ds = ds.map(rate_translation, num_proc=4)

    ds.push_to_hub(
        "flozi00/translation-quality-german-english",
        config_name=DS_NAME.replace("/", "-"),
    )
