import datasets
from tqdm import tqdm
from TOKENS import *

modes = []
all_rows = []

ds = datasets.load_dataset(
    "argilla/databricks-dolly-15k-curated-multilingual", split="de+en"
)
for row in tqdm(ds, desc="Databricks Dolly"):
    all_rows.append(f'{row["context"]}\n{row["instruction"]}')
    modes.append(row["category"])

ds = datasets.Dataset.from_dict({"text": all_rows, "label": modes})

ds.push_to_hub("LLM-Task-Classification")
