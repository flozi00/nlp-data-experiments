from transformers import pipeline
import torch
import tqdm

pipe = pipeline(
    "text-classification",
    model="flozi00/multilingual-e5-large-llm-tasks",
    torch_dtype=torch.float16,
    model_kwargs={
        "load_in_4bit": True,
    },
)


def get_dolly_label(prompt: list) -> list:
    labels = []
    for i in tqdm.tqdm(range(0, len(prompt), 16)):
        try:
            category = pipe(prompt[i : i + 16], batch_size=16, truncation=True)
            labels.extend([x["label"] for x in category])
        except Exception as e:
            print(e)
    return labels
