# src/02_make_text_embeddings_transformers.py
import os, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

META = "data/raw/final_multimodal_dataset.csv"
OUT_DIR = "outputs/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_EMB = os.path.join(OUT_DIR, "text_embeddings.npy")
OUT_KEYS = os.path.join(OUT_DIR, "text_keys.npy")

MODEL = "sentence-transformers/all-mpnet-base-v2"

print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to("cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

df = pd.read_csv(META)
texts = (df["findings"].fillna("") + " " + df["impression"].fillna("")).tolist()

batch_size = 8
embs = []

print("Encoding text...")
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    embs.append(pooled.cpu().numpy())

embs = np.vstack(embs).astype("float32")
np.save(OUT_EMB, embs)
np.save(OUT_KEYS, df["filename"].to_numpy())

print("Saved text embeddings:", OUT_EMB)
