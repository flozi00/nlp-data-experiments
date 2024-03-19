import datasets
from unidecode import unidecode
from difflib import SequenceMatcher
from nemo.collections.asr.models import EncDecMultiTaskModel
import torch_tensorrt
import torch


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() >= 0.95


# load model
canary_model = EncDecMultiTaskModel.from_pretrained(
    "nvidia/canary-1b", map_location="cuda:0"
)

canary_model = torch.compile(
    canary_model, mode="max-autotune", backend="torch_tensorrt", fullgraph=True
)


# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)


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
    "mozilla-foundation/common_voice_16_1",
    "de",
    split="train+test+validation",
    cache_dir="volume_ds_cache",
)
cv = cv.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=False))
cv = cv.rename_column("sentence", "transkription")

# Filter out the data
cv = cv.filter(lambda x: x["up_votes"] >= 2 and x["down_votes"] == 0)
cv = cv.filter(lambda x: len(x["transkription"]) > 10 and len(x["transkription"]) < 512)

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
    lambda example: len(example["transkription"]) > 10
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
    lambda example: len(example["transkription"]) > 10
    and len(example["transkription"]) < 512
)
features = list(mls.column_names)
for feature in features:
    if feature != "transkription" and feature != "audio":
        mls = mls.remove_columns(feature)

new_column = ["multilingual librispeech"] * len(mls)
mls = mls.add_column("source", new_column)

voxpopuli = datasets.concatenate_datasets([voxpopuli, mls])

print(voxpopuli)

audios = [voxpopuli[i]["audio"]["path"] for i in range(len(voxpopuli))]
with torch.autocast(enabled=True, dtype=torch.float16, device_type="cuda"):
    with torch.inference_mode():
        transcript = canary_model.transcribe(audios, batch_size=96)
        for i in range(len(transcript)):
            transcript[i] = (
                transcript[i]
                .replace(" ,", ",")
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ' ", "'")
                .replace(" - ", "-")
                .replace(" : ", ":")
                .replace(" ; ", ";")
                .replace(" %", "%")
            )

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

cv.save_to_disk("asr-german-canary")

cv.push_to_hub("flozi00/asr-german-canary")
