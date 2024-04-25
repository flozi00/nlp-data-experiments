import pysrt
import os
from tqdm import tqdm
import json
import librosa
import soundfile as sf

prefix = "7609244"
output_folder = "audio_output"
sampling_rate = 16000

subs = pysrt.open(f"{prefix}.srt")
subs.shift(seconds=-2)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


result = []
text = ""
prev_end = 0
prev_start = 0

speech = librosa.load(f"{prefix}_mp3_128kb_stereo_de_128.mp3", sr=sampling_rate)[0]

for sub in subs:
    start = (
        sub.start.seconds
        + sub.start.minutes * 60
        + sub.start.hours * 3600
        + sub.start.milliseconds / 1000
    )
    end = (
        sub.end.seconds
        + sub.end.minutes * 60
        + sub.end.hours * 3600
        + sub.end.milliseconds / 1000
    )

    if prev_start == 0:
        prev_start = start

    if prev_end == 0:
        prev_end = end

    if start - prev_end >= 5:
        if text == "":
            text = sub.text
        result.append(
            {
                "start": int(prev_start - 3.5),
                "end": int(prev_end + 1.5),
                "text": text.replace("\n", " "),
            }
        )
        prev_start = 0
        prev_end = end
        text = sub.text
    else:
        text += sub.text + " "
        prev_end = end

for res in tqdm(result):
    if res["end"] - res["start"] < 30:
        start = res["start"] * sampling_rate
        end = res["end"] * sampling_rate
        text = res["text"]
        part = speech[start:end]
        sf.write(
            f"{output_folder}/{prefix}_{res['start']}_{res['end']}.wav",
            part,
            sampling_rate,
        )
        with open(f"{output_folder}/transkript.jsonl", "a", encoding="utf-8") as f:
            f.write(
                f'{json.dumps({"start": res["start"], "end": res["end"], "text": text})})\n'
            )
