import datasets
from TOKENS import *
from datas.bactrian import bactrian
from datas.dolly import dolly
from datas.evolinstruct import evol
from datas.openassistant import oa
from datas.belebele import belebele
from datas.germandpr import germandpr
from utils.uncensore_phrases import PHRASES

sets = [dolly, bactrian, evol, oa, belebele, germandpr]


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    all_labels = []
    from_ds = []

    for dset in sets:
        results = dset()
        all_rows.extend(results[0])
        all_labels.extend(results[1])
        from_ds.extend(results[2])

    ds = datasets.Dataset.from_dict(
        {
            "conversations": all_rows,
            "from": from_ds,
            "labels": all_labels,
        }
    )

    return ds


final_data = get_chat_dataset()

print(final_data)

for phrase in PHRASES:
    final_data = final_data.filter(
        lambda x: phrase.lower() not in x["conversations"].lower()
    )

final_data = final_data.filter(lambda x: x["labels"] != "error")

print(final_data)

final_data.push_to_hub("conversations", max_shard_size="1GB")
