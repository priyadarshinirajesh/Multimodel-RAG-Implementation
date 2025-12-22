# src/04_build_faiss_text_flava.py

import numpy as np
import faiss
import os

IN_EMB = "outputs/embeddings/text_embeddings.npy"
OUT_INDEX = "outputs/embeddings/faiss_text.index"

emb = np.load(IN_EMB).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
faiss.normalize_L2(emb)

index.add(emb)
faiss.write_index(index, OUT_INDEX)

print("FAISS Text Index Saved:", OUT_INDEX)
print("Total vectors indexed:", emb.shape[0])
