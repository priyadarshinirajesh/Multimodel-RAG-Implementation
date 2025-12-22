# src/03_make_text_embeddings_flava.py

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

META = "data/raw/final_multimodal_dataset.csv"
OUT_DIR = "outputs/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EMB = os.path.join(OUT_DIR, "text_embeddings.npy")
OUT_KEYS = os.path.join(OUT_DIR, "text_keys.npy")

model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cpu")

df = pd.read_csv(META)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

texts = (
    df["findings"].fillna("") + " " +
    df["impression"].fillna("") + " " +
    df["indication"].fillna("") + " " +
    df["comparison"].fillna("")
).tolist()

embs = []
batch = 16

for i in tqdm(range(0, len(texts), batch)):
    batch_text = texts[i:i+batch]
    enc = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    embs.append(pooled.cpu().numpy())

embs = np.vstack(embs).astype("float32")

np.save(OUT_EMB, embs)
np.save(OUT_KEYS, df["filename"].values)

print("\n✔ Saved text embeddings:", OUT_EMB)
print("Total rows:", len(df))
