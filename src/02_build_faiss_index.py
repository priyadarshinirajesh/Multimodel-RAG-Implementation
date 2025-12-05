# src/02_build_faiss_index.py
import numpy as np
import faiss
import os

EMB_PATH = "outputs/embeddings/image_embeddings.npy"
KEY_PATH = "outputs/embeddings/image_keys.npy"
OUT_INDEX = "outputs/embeddings/faiss_image.index"

embs = np.load(EMB_PATH).astype("float32")

# FAISS index – using cosine similarity (normalized already)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, OUT_INDEX)

print("FAISS Index Saved:", OUT_INDEX)
print("Total vectors indexed:", index.ntotal)
