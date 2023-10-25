from tqdm import tqdm
from TOKENS import BOT, PROMPTER, END

from utils.detector import detector
from utils.classifier import get_dolly_label


def process_3_part_ds(
    first,
    second,
    output,
    data,
) -> tuple[list, list]:
    ds = []
    labels = []
    for row in tqdm(data):
        if detector(row[first] + row[second]) == detector(row[output]) == "de":
            ds.append(
                f"{PROMPTER}{row[first]}\n{row[second]}{END}{BOT}{row[output]}{END}"
            )
            try:
                labels.append(row["category"])
            except Exception:
                labels.append(get_dolly_label(f"{row[first]}\n{row[second]}"))

    return ds, labels
