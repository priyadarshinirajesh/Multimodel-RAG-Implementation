# src/02_build_faiss_index_image_flava.py
import numpy as np
import faiss
import os

EMB_PATH = "outputs/embeddings/image_embeddings_flava.npy"
KEY_PATH = "outputs/embeddings/image_keys_flava.npy"
OUT_INDEX = "outputs/embeddings/faiss_image_flava.index"

embs = np.load(EMB_PATH).astype("float32")

print("[DEBUG] Loaded embeddings shape:", embs.shape)

# 🔒 Safety: ensure normalization
faiss.normalize_L2(embs)

# Cosine similarity index
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, OUT_INDEX)

print("✅ FAISS Index Saved:", OUT_INDEX)
print("✅ Total vectors indexed:", index.ntotal)
print("✅ Embedding dimension:", embs.shape[1])
