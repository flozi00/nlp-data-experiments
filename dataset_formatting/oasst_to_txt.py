import datasets

ds = datasets.load_dataset(
    "flozi00/openassistant-oasst1-flattened-filtered", split="train"
)

dslangs = list(set(ds["lang"]))

dslangs = ["en"]

for lang in dslangs:
    ds1 = ds.filter(lambda example: example["lang"] == lang)
    print(ds1)
    ds1 = ds1.filter(lambda example: "import " not in example["conversations"])
    ds1 = ds1.filter(lambda example: "```" not in example["conversations"])
    ds1 = ds1.filter(lambda example: "#include" not in example["conversations"])
    ds1 = ds1.filter(lambda example: len(example["conversations"]) < 4096)
    ds1 = ds1.remove_columns("lang")
    print(ds1)

    stepsize = 10000
    for slice in range(0, len(ds1["conversations"]), stepsize):
        txt = "\n***\n".join(ds1["conversations"][slice : slice + stepsize]).replace(
            "\n\n", "\n"
        )

        with open(f"_oasst_{lang}_{slice}.txt", "w+") as myfile:
            myfile.write(txt)
