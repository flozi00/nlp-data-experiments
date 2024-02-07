import datasets
from TOKENS import *
from datas.dolly import dolly
from datas.evolinstruct import evol
from datas.openassistant import oa
from datas.belebele import belebele
from datas.bactrian import bactrian
from datas.no_robots_german import no_robots
from datas.alpaca_gpt4 import alpaca
from datas.function_calling import function_calling
from datas.self_instruct_gpt4 import self_instruct_gpt4
from datas.ultra_chat import ultra_chat
from datas.wiki_qa import wiki_qa
from utils.format import convert_to_sharegpt
from utils.uncensore_phrases import PHRASES

labeled_sets = [
    oa,
    belebele,
    bactrian,
    evol,
    alpaca,
    no_robots,
    dolly,
    function_calling,
    self_instruct_gpt4,
    ultra_chat,
    wiki_qa,
]


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    labels = []

    for dset in labeled_sets:
        results = dset()
        all_rows.extend(results[0])
        from_ds.extend(results[1])
        labels.extend(results[2])

    for i in range(len(all_rows)):
        all_rows[i] = str(all_rows[i]).strip()

    ds = datasets.Dataset.from_dict(
        {
            "conversations": all_rows,
            "from": from_ds,
            "labels": labels,
            "sharegpt": convert_to_sharegpt(all_rows),
        }
    )

    return ds


final_data = get_chat_dataset()

print(final_data)

for phrase in PHRASES:
    final_data = final_data.filter(
        lambda x: phrase.lower() not in x["conversations"].lower()
    )

final_data = final_data.filter(lambda x: len(x["conversations"]) >= 128)

print(final_data)

from_labels = final_data.unique("from")
labeling = datasets.ClassLabel(names=from_labels)
final_data.cast_column("from", labeling)

final_data.push_to_hub("flozi00/conversations", max_shard_size="1GB")
