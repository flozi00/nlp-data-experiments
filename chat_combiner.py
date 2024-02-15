import datasets
from TOKENS import PROMPTER, BOT, SYSTEM, END
from datas.dolly import dolly
from datas.openassistant import oa
from datas.belebele import belebele
from datas.no_robots_german import no_robots
from datas.function_calling import function_calling
from utils.format import convert_to_sharegpt
from utils.classifier import get_dolly_label
from utils.uncensore_phrases import PHRASES

labeled_sets = [
    oa,
    belebele,
    no_robots,
    dolly,
    function_calling,
]


def get_chat_dataset() -> datasets.Dataset:
    all_rows = []
    from_ds = []
    labels = []

    for dset in labeled_sets:
        results = dset()
        for i in range(len(results[0])):
            if results[0][i].count(PROMPTER) == results[0][i].count(BOT) >= 1:
                all_rows.append(results[0][i])
                from_ds.append(results[1][i])
                labels.append(results[2][i])

    conversations = convert_to_sharegpt(all_rows)

    first_messages = []
    first_answer = []

    # get the first message from human of each conversation
    for conv in conversations:
        human_found = False
        for msg in conv:
            if msg["from"] == "human" and not human_found:
                first_messages.append(msg["value"])
                human_found = True
        if not human_found:
            print(conv)
            exit()

    # get the first answer from the AI of each conversation
    for conv in conversations:
        bot_found = False
        for msg in conv:
            if msg["from"] == "gpt" and not bot_found:
                first_answer.append(msg["value"])
                bot_found = True
        if not bot_found:
            print(conv)
            exit()

    # check if all labels are present
    for i in range(len(all_rows)):
        if labels[i] == "unknown":
            labels[i] = get_dolly_label(first_messages[i])

    ds = datasets.Dataset.from_dict(
        {
            "raw": all_rows,
            "from": from_ds,
            "labels": labels,
            "conversations": conversations,
            "first_message": first_messages,
            "first_answer": first_answer,
        }
    )

    return ds


final_data = get_chat_dataset()

print(final_data)

for phrase in PHRASES:
    final_data = final_data.filter(lambda x: phrase.lower() not in x["raw"].lower())

final_data = final_data.filter(lambda x: len(x["raw"]) >= 128)

print(final_data)

from_labels = final_data.unique("from")
labeling = datasets.ClassLabel(names=from_labels)
final_data.cast_column("from", labeling)

final_data.push_to_hub("flozi00/conversations", max_shard_size="1GB")
