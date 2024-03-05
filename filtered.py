import datasets

ds = datasets.load_dataset("flozi00/translation-quality-german-english", split="train")

ds.filter(lambda x: x["label"] == "Good").to_csv("de_en.csv")

ds.filter(lambda x: x["label"] in ["Good", "Bad", "Neutral"]).to_csv("de_en_all.csv")
