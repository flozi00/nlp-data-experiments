import datasets

ger = []
en = []

alma = datasets.load_dataset("haoranxu/ALMA-Human-Parallel", "de-en", split="train")

dolly = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de"
)

for i in alma["translation"]:
    if len(i["de"]) >= 32 and len(i["en"]) >= 32:
        ger.append(i["de"])
        en.append(i["en"])

for i in dolly:
    if len(i["instruction"]) >= 16 and len(i["instruction_original_en"]) >= 16:
        ger.append(i["instruction"])
        en.append(i["instruction_original_en"])
    if len(i["context"]) >= 16 and len(i["context_original_en"]) >= 16:
        ger.append(i["context"])
        en.append(i["context_original_en"])

    if len(i["response"]) >= 16 and len(i["response_original_en"]) >= 16:
        ger.append(i["response"])
        en.append(i["response_original_en"])


ds = datasets.Dataset.from_dict({"de": ger, "en": en})

ds.push_to_hub("de_en_parallel_filtered")
