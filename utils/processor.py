from tqdm import tqdm
from TOKENS import BOT, PROMPTER, END

from utils.detector import detector


def process_3_part_ds(
    first,
    second,
    output,
    data,
) -> tuple[list, list]:
    ds = []
    for row in tqdm(data):
        if detector(row[first] + row[second]) == detector(row[output]) == "de":
            ds.append(
                f"{PROMPTER}{row[first]}\n{row[second]}{END}{BOT}{row[output]}{END}"
            )

    return ds
