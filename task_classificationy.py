import datasets
from tqdm import tqdm
from TOKENS import *

modes = []
all_rows = []

ds = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de+en+es+fr"
)

labels = ds.unique("category")

LABELS_TO_IDS = {label: i for i, label in enumerate(labels)}
IDS_TO_LABELS = {i: label for i, label in enumerate(labels)}

for row in tqdm(ds, desc="Databricks Dolly"):
    all_rows.append(f'{row["context"]}\n{row["instruction"]}')
    modes.append(row["category"])

ds = datasets.Dataset.from_dict({"text": all_rows, "label": modes})

smallest_category_count = int(1e9)

new_ds = []
for mode in labels:
    count = len(ds.filter(lambda example: example["label"] == mode))
    if count < smallest_category_count:
        smallest_category_count = count

for mode in labels:
    new_ds.append(
        ds.filter(lambda example: example["label"] == mode)
        .shuffle()
        .select(range(smallest_category_count - 1))
    )

ds: datasets.Dataset = datasets.concatenate_datasets(new_ds)

ds.push_to_hub("LLM-Task-Classification", config_name="multilingual")
