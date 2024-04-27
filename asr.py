import datasets
from unidecode import unidecode
from difflib import SequenceMatcher
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
import torch_tensorrt  # noqa
from tqdm.auto import tqdm
import re

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model_id = "primeline/whisper-large-v3-german"
#model_id = "primeline/distil-whisper-large-v3-german"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

model = torch.compile(
    model, mode="max-autotune", backend="torch_tensorrt", fullgraph=True
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=64,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)


def similar(a, b):
    a = re.sub(r'[^\w\s]', '', a)
    b = re.sub(r'[^\w\s]', '', b)
    return SequenceMatcher(None, a, b).ratio() >= 0.95


def normalize_text(batch):
    text = batch["transkription"]
    couples = [
        ("ä", "ae"),
        ("ö", "oe"),
        ("ü", "ue"),
        ("Ä", "Ae"),
        ("Ö", "Oe"),
        ("Ü", "Ue"),
    ]

    # Replace special characters with their ascii equivalent
    for couple in couples:
        text = text.replace(couple[0], f"__{couple[1]}__")
    text = text.replace("ß", "ss")
    text = unidecode(text)

    # Replace the ascii equivalent with the original character after unidecode
    for couple in couples:
        text = text.replace(f"__{couple[1]}__", couple[0])

    batch["transkription"] = text
    return batch


# Doing all the commonvoice related filtering stuff
cv = datasets.load_dataset(
    "fsicoli/common_voice_17_0",
    "de",
    split="train+test+validation",
    cache_dir="volume_ds_cache",
)
cv = cv.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=False))
cv = cv.rename_column("sentence", "transkription")

# Filter out the data
cv = cv.filter(lambda x: x["up_votes"] >= 2 and x["down_votes"] == 0)
cv = cv.filter(lambda x: len(x["transkription"]) > 32 and len(x["transkription"]) < 512)

# Remove all columns not needed
features = list(cv.column_names)
for feature in features:
    if feature != "transkription" and feature != "audio":
        cv = cv.remove_columns(feature)

new_column = ["cv_16_1"] * len(cv)
cv = cv.add_column("source", new_column)

# do the voxpopuli stuff
voxpopuli = datasets.load_dataset(
    "facebook/voxpopuli",
    "de",
    split="train+test+validation",
    cache_dir="volume_ds_cache",
)
voxpopuli = voxpopuli.cast_column(
    "audio", datasets.Audio(sampling_rate=16000, decode=False)
)
voxpopuli = voxpopuli.rename_column("raw_text", "transkription")
voxpopuli = voxpopuli.filter(lambda example: example["is_gold_transcript"] is True)
voxpopuli = voxpopuli.filter(
    lambda example: len(example["transkription"]) > 32
    and len(example["transkription"]) < 512
)
features = list(voxpopuli.column_names)
for feature in features:
    if feature != "transkription" and feature != "audio":
        voxpopuli = voxpopuli.remove_columns(feature)

new_column = ["voxpopuli"] * len(voxpopuli)
voxpopuli = voxpopuli.add_column("source", new_column)


mls = datasets.load_dataset(
    "facebook/multilingual_librispeech",
    "german",
    split="train+test+validation",
    cache_dir="volume_ds_cache",
)
mls = mls.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=False))
mls = mls.rename_column("text", "transkription")
mls = mls.filter(
    lambda example: len(example["transkription"]) > 32
    and len(example["transkription"]) < 512
)
features = list(mls.column_names)
for feature in features:
    if feature != "transkription" and feature != "audio":
        mls = mls.remove_columns(feature)

new_column = ["multilingual librispeech"] * len(mls)
mls = mls.add_column("source", new_column)

voxpopuli = datasets.concatenate_datasets([voxpopuli, mls])

print(cv, voxpopuli)

audios = [voxpopuli[i]["audio"]["path"] for i in range(len(voxpopuli))]
transcript = []
with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
    with torch.inference_mode():
        batch_size = 16
        for i in tqdm(range(0, len(audios), batch_size)):
            batch_audios = audios[i:i+batch_size]
            batch_results = pipe(batch_audios, generate_kwargs={"language": "de", "task": "transcribe"},)
            for result in batch_results:
                #print(result)
                transcript.append(result["text"].strip())

# add the transcriptions to the dataset and filter
voxpopuli = voxpopuli.add_column("canary_labels", transcript)

voxpopuli = voxpopuli.filter(
    lambda example: similar(
        example["transkription"].lower(), example["canary_labels"].lower()
    )
)
print(voxpopuli)

voxpopuli = voxpopuli.remove_columns(["transkription"]).rename_column(
    "canary_labels", "transkription"
)


cv: datasets.Dataset = datasets.concatenate_datasets([cv, voxpopuli])

# start the quality filtering
cv = cv.map(normalize_text)

print(cv)

cv = cv.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=True))
cv.save_to_disk("asr-german-mixed")
cv.push_to_hub("flozi00/asr-german-mixed")
