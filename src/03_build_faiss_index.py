# src/03_build_faiss_index.py
import os, numpy as np, faiss
OUT_DIR = "outputs/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

for name in ["image", "text"]:
    emb = np.load(f"outputs/embeddings/{name}_embeddings.npy").astype("float32")
    # ensure normalized
    faiss.normalize_L2(emb)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, os.path.join(OUT_DIR, f"faiss_{name}.index"))
    print("Saved", os.path.join(OUT_DIR, f"faiss_{name}.index"))
