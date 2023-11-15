import datasets

with open("data.txt", "r") as f:
    txt = f.read()

txt = txt.split("\n***\n")

ds = datasets.Dataset.from_dict({"messages": txt})

ds.push_to_hub("no_robots_german")