from transformers import pipeline
import torch
import tqdm
import torch_tensorrt

pipe = pipeline(
    "text-classification",
    model="flozi00/multilingual-e5-large-llm-tasks",
    device=0,
    # torch_dtype=torch.float16,
    # model_kwargs = {"attn_implementation":"sdpa",},
)
pipe.model.eval()
# pipe.model = torch.compile(pipe.model, mode="reduce-overhead")


from filecache import filecache


@filecache(7 * 24 * 60 * 60)
def get_dolly_label(prompt: str) -> str:
    category = pipe(prompt, truncation=True)[0]["label"]
    return category
