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
    "text2text-generation",
    model="flozi00/t5-small-llm-tasks",
    device=0,
    torch_dtype=torch.float16,
)
pipe.model = BetterTransformer.transform(pipe.model)


@filecache(7 * 24 * 60 * 60)
def get_dolly_label(prompt: str) -> str:
    return pipe(
        f"Labels: closed_qa, classification, open_qa, information_extraction, brainstorming, general_qa, summarization, creative_writing </s> Input: {prompt}",
        max_new_tokens=5,
        do_sample=False,
    )[0]["generated_text"]
