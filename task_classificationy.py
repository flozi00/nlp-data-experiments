import datasets
from tqdm import tqdm

modes = []
searchable = []
all_rows = []
named_modes = []

ds = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de+en+es+fr"
)

for row in tqdm(ds, desc="Databricks Dolly"):
    all_rows.append(f'{row["context"]}\n{row["instruction"]}')
    named_modes.append(row["category"])


ds = datasets.load_dataset(
    "flozi00/classify-llm-tasks-german",
    split="train",
)

for row in tqdm(ds, desc="Flozi"):
    all_rows.append(row["input"])
    named_modes.append(row["output"])


ds = datasets.Dataset.from_dict(
    {
        "text": all_rows,
        "named_labels": named_modes,
    }
)


ds.push_to_hub("LLM-Task-Classification", config_name="multilingual")
