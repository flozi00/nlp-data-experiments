from nomic import atlas
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# make dataset
max_documents = 1024
dataset = load_dataset("flozi00/conversations")["train"]
documents = [
    dataset[i] for i in np.random.randint(len(dataset), size=max_documents).tolist()
]


initial = True

with torch.no_grad():
    batch = [document["conversations"] for document in documents]
    for x in documents:
        del x["conversations"]
    embeddings_batch = model.encode(
        batch, normalize_embeddings=True, batch_size=512, show_progress_bar=True
    )

    project = atlas.map_embeddings(
        embeddings=embeddings_batch,
        data=documents,
        topic_label_field="title",
        name="A Map That Gets Updated",
        reset_project_if_exists=True,
        add_datums_if_exists=False,
    )
    map = project.get_map("A Map That Gets Updated")
    print(map)
    initial = False
