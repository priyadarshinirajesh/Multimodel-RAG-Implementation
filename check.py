import numpy as np, pandas as pd
df = pd.read_csv("data/raw/final_multimodal_dataset.csv")
emb = np.load("outputs/embeddings/image_embeddings.npy")
keys = np.load("outputs/embeddings/image_keys.npy", allow_pickle=True)
print("CSV rows:", len(df))
print("Saved image embeddings:", emb.shape, "count keys:", len(keys))
# Quick mismatch check
if len(df) != emb.shape[0] or len(df) != len(keys):
    print("⚠️ Count mismatch between CSV and saved embeddings")
else:
    print("OK: embeddings match CSV row count")
