from filecache import filecache
from langdetect import detect
from transformers import pipeline
from optimum.bettertransformer import BetterTransformer
import torch
from filecache import filecache


@filecache(7 * 24 * 60 * 60)
def detector(text: str) -> str:
    try:
        return detect(text)
    except:
        return None


pipe = pipeline(
    "text-classification",
    model="flozi00/multilingual-e5-large-llm-tasks",
    device=0,
    torch_dtype=torch.float16,
)
pipe.model = BetterTransformer.transform(pipe.model)


@filecache(7 * 24 * 60 * 60)
def get_dolly_label(prompt: str) -> str:
    return pipe(prompt)[0]["label"].strip()
