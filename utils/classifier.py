from filecache import filecache
from transformers import pipeline
import torch

pipe = pipeline(
    "text-classification",
    model="flozi00/multilingual-e5-large-llm-tasks",
    device=0,
    torch_dtype=torch.float16,
)


@filecache(7 * 24 * 60 * 60)
def get_dolly_label(prompt: str) -> str:
    prompt = prompt[: 450 * 3]
    try:
        category = pipe(prompt)[0]["label"].strip()
    except Exception as e:
        print(e)
        print(prompt)
        category = "error"
    return category
