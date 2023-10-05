import datasets
from tqdm import tqdm
from TOKENS import *
from datasets import ClassLabel

modes = []
searchable = []
all_rows = []

ds = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
)

labels = ds.unique("category")

labeling = ClassLabel(names=labels)
search_labeling = ClassLabel(num_classes=2, names=["not_searchable", "searchable"])

LABELS_TO_IDS = {label: i for i, label in enumerate(labels)}
IDS_TO_LABELS = {i: label for i, label in enumerate(labels)}

for row in tqdm(ds, desc="Databricks Dolly"):
    all_rows.append(f'{row["context"]}\n{row["instruction"]}')
    modes.append(row["category"])
    searchable.append(1 if row["category"] == "open_qa" else 0)

ds = datasets.Dataset.from_dict(
    {"text": all_rows, "label": modes, "searchable": searchable}
)

ds.cast_column("label", labeling)
ds.cast_column("searchable", search_labeling)

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
