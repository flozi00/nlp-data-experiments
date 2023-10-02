from utils.processor import process_3_part_ds
import datasets


def bactrian():
    all_rows = []
    all_labels = []
    from_ds = []
    """
    The Bactrian-X dataset is a collection of 3.4M instruction-response pairs in 52 languages, 
    that are obtained by translating 67K English instructions (alpaca-52k + dolly-15k) into 51 languages using Google Translate API. 
    The translated instructions are then fed to ChatGPT (gpt-3.5-turbo) to obtain its natural responses, 
    resulting in 3.4M instruction-response pairs in 52 languages (52 languages x 67k instances = 3.4M instances).
    """
    ds = datasets.load_dataset("MBZUAI/Bactrian-X", "de", split="train")
    ds_processed, labels_processed = process_3_part_ds(
        "instruction",
        "input",
        "output",
        ds,
    )
    all_rows.extend(ds_processed)
    all_labels.extend(labels_processed)
    from_ds.extend(["MBZUAI/Bactrian-X"] * len(ds_processed))

    return all_rows, all_labels, from_ds
