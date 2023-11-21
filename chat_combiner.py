import datasets
from TOKENS import *
from datas.bactrian import bactrian
from datas.dolly import dolly
from datas.evolinstruct import evol
from datas.openassistant import oa
from datas.belebele import belebele
from datas.germandpr import germandpr, germandpr_rag
from datas.no_robots_german import no_robots
from datas.schnabeltier import schnabeltier
from datas.germanpoems import german_poems
from datas.germansongs import german_songs
from datas.germanqa import germanqa
from datas.single_queries import single_queries
from utils.uncensore_phrases import PHRASES

labeled_sets = [
    # oa,
    belebele,
    germandpr,
    germandpr_rag,
    # bactrian,
    # evol,
    no_robots,
    dolly,
    # schnabeltier,
    german_poems,
    german_songs,
    # germanqa,
    single_queries,
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

final_data.push_to_hub("conversations", max_shard_size="1GB")
