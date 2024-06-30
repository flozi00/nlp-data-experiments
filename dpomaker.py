import datasets
from dpo_maker.belebele import belebele
from dpo_maker.distilabel_math import math_dpo
from dpo_maker.sentiment_mteb import sentiment_mteb_dpo
from dpo_maker.mtop_domain import mtop_domain

classification_data = [belebele(), sentiment_mteb_dpo(), mtop_domain()]
qa_data = [belebele()]
_math = [math_dpo()]


def transform_list_to_dpo(data: list) -> datasets.Dataset:
    prompt = []
    chosen = []
    rejected = []
    for d in data:
        prompt.extend(d[0])
        chosen.extend(d[1])
        rejected.extend(d[2])

    ds = datasets.Dataset.from_dict(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    return ds


final_dataset = datasets.DatasetDict()
final_dataset["classification"] = transform_list_to_dpo(classification_data)
final_dataset["qa"] = transform_list_to_dpo(qa_data)
final_dataset["math"] = transform_list_to_dpo(_math)

final_dataset.push_to_hub("DPO_Tasks_Curated", private=True)
