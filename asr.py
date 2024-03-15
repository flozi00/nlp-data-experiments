import datasets
from gradio_client import Client, file
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() >= 0.95


voxpopuli = datasets.load_dataset("facebook/voxpopuli", "de", split="train")
client = Client("http://localhost:7860/")


def add_pseudo_labels(example):
    example["transcription"] = (
        client.predict(
            file(example["audio"]["path"]),
            "German",
            "German",
            True,
            api_name="/transcribe",
        )
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
    )
    return example


voxpopuli = voxpopuli.map(add_pseudo_labels)

voxpopuli = voxpopuli.filter(
    lambda example: similar(
        example["raw_text"].lower(), example["transcription"].lower()
    )
)

voxpopuli = voxpopuli.remove_columns(
    [
        "audio_id",
        "language",
        "raw_text",
        "is_gold_transcript",
        "normalized_text",
        "gender",
        "speaker_id",
        "accent",
    ]
)

voxpopuli.save_to_disk("voxpopuli_filtered_de")

voxpopuli.push_to_hub("flozi00/voxpopuli_filtered_de", max_shard_size="5GB")
