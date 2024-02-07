import datasets
from tqdm import tqdm
from TOKENS import BOT, PROMPTER, END



def process_3_part_ds(
    first,
    second,
    output,
    data,
) -> tuple[list, list]:
    ds = []
    for row in tqdm(data):
        ds.append(
            f"{PROMPTER}{row[first]}\n{row[second]}{END}{BOT}{row[output]}{END}"
        )

    return ds


def bactrian() -> tuple[list, list, list]:
    all_rows = []
    from_ds = []
    labels = []
    """
    The Bactrian-X dataset is a collection of 3.4M instruction-response pairs in 52 languages, 
    that are obtained by translating 67K English instructions (alpaca-52k + dolly-15k) into 51 languages using Google Translate API. 
    The translated instructions are then fed to ChatGPT (gpt-3.5-turbo) to obtain its natural responses, 
    resulting in 3.4M instruction-response pairs in 52 languages (52 languages x 67k instances = 3.4M instances).
    """
    ds = datasets.load_dataset("MBZUAI/Bactrian-X", "de", split="train")
    ds_processed = process_3_part_ds(
        "instruction",
        "input",
        "output",
        ds,
    )
    all_rows.extend(ds_processed)
    from_ds.extend(["MBZUAI/Bactrian-X"] * len(ds_processed))
    labels.extend(["unknown"] * len(ds_processed))

    return all_rows, from_ds, labels
