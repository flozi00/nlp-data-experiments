import os

import datasets
import torch
from transformers import ModernBertForSequenceClassification, pipeline

_GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0")


def load_model(gpu_index=0):
    model = ModernBertForSequenceClassification.from_pretrained(
        "flozi00/GermanEduScorer-ModernBERT-base",
        reference_compile=False,
        attn_implementation="sdpa",
    ).to(torch.bfloat16)

    model = torch.compile(model, dynamic=True, mode="max-autotune")

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer="flozi00/GermanEduScorer-ModernBERT-base",
        device=gpu_index,
        torch_dtype=torch.bfloat16,
    )

    return pipe


pipe0 = load_model(0)
tokenizer_kwargs = {"truncation": True}

BAD_WORDS = [
    "Sofort lieferbar",
]


def process_chunk(pipe, texts):
    if not texts:
        return []
    return [
        int(x["label"])
        for x in pipe(
            texts,
            batch_size=256,
            truncation=True,
            max_length=1024,
        )
    ]


def classification_wrapper(text_list: list):
    return process_chunk(pipe0, text_list)


def map_edu(example):
    example["content"] = example["text"]
    example["label"] = classification_wrapper(example["text"])
    return example


for SET_ID in ["0", "1", "2", "3"]:
    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/deu_Latn/train/"
    data_files = {
        "train": [base_url + f"00{SET_ID}_0000{i}.parquet" for i in range(10)]
        + [base_url + f"00{SET_ID}_000{i}.parquet" for i in range(10, 38)]
    }

    fineweb = datasets.load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        num_proc=4,
        cache_dir=f"./cache_fineweb_{SET_ID}",
    )

    chunk_size = 100_000
    part_size = len(fineweb) // 4
    total_samples = part_size * (int(_GPU_ID) + 1)
    output_path = f"fineweb2_edu_4up_german_split_{int(SET_ID)+1}-of-4"

    for i in range(part_size * int(_GPU_ID), total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        checkpoint_path = f"chunks/{output_path}_chunk_{i}"

        # Try to load existing chunk
        try:
            dset = datasets.load_from_disk(checkpoint_path)
            print(f"Chunk {i} to {end_idx} already processed, skipping...")
            continue
        except Exception:
            print(f"Processing chunk {i} to {end_idx} of {total_samples}")

            chunk = fineweb.select(range(i, end_idx))
            processed_chunk = chunk.map(
                map_edu,
                remove_columns=chunk.column_names,
                batch_size=1024,
                batched=True,
            ).filter(lambda x: x["label"] >= 4, num_proc=8)
            processed_chunk = processed_chunk.rename_column("content", "text")

            processed_chunk.save_to_disk(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if i % 1_000_000 == 0 and _GPU_ID == "0" and i > 0:
            sets_to_push = []
            # list all folders in the chunks directory
            for folder in os.listdir("chunks"):
                # load the dataset
                sets_to_push.append(datasets.load_from_disk(f"chunks/{folder}"))
            state_ds = datasets.concatenate_datasets(sets_to_push)
            for bad_word in BAD_WORDS:
                state_ds = state_ds.filter(
                    lambda x: bad_word not in x["text"], num_proc=8
                )
            state_ds = state_ds.filter(
                lambda x: len(x["text"]) > 1024 and len(x["text"]) <= 100_000,
                num_proc=8,
            )
            state_ds.push_to_hub("Fineweb2-German-Eduscore-4andMore")
