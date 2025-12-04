# src/04_retrieve_and_rerank.py
import os
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# --------- CONFIG ---------
META = "data/raw/final_multimodal_dataset.csv"
IMG_INDEX_PATH = "outputs/faiss/faiss_image.index"
TXT_INDEX_PATH = "outputs/faiss/faiss_text.index"
IMG_KEYS = "outputs/embeddings/image_keys.npy"
TXT_KEYS = "outputs/embeddings/text_keys.npy"

# Text embedding model (must match the model used to create text embeddings)
TEXT_EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"  # we used this via transformers pooling earlier
DEVICE = "cpu"
TOP_K = 5

# --------- UTIL: pooling (same as in embedder) ---------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --------- LOAD models / indexes / keys ---------
print("Loading tokenizer/model for query encoding...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_EMB_MODEL)
model = AutoModel.from_pretrained(TEXT_EMB_MODEL).to(DEVICE)

print("Loading FAISS indexes and keys...")
img_index = faiss.read_index(IMG_INDEX_PATH)
txt_index = faiss.read_index(TXT_INDEX_PATH)
img_keys = np.load(IMG_KEYS, allow_pickle=True)
txt_keys = np.load(TXT_KEYS, allow_pickle=True)

meta = pd.read_csv(META)

# --------- QUERY ENCODER ---------
def encode_query(text):
    enc = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    q_emb = pooled.cpu().numpy().astype("float32")
    return q_emb

# --------- SEARCH helpers ---------
def search_index(index, q_emb, k=TOP_K):
    # index expects normalized vectors if IndexFlatIP used with normalized embeddings
    # ensure q_emb is normalized
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0], I[0]

def pretty_print_results(filenames, scores, source):
    print(f"\nTop {len(filenames)} results from {source}:")
    for fname, sc in zip(filenames, scores):
        # get row in meta if exists
        row = meta[meta["filename"] == fname]
        pid = row["patient_id"].values[0] if len(row) else "N/A"
        modality = row["modality"].values[0] if len(row) else "N/A"
        print(f"- {fname}  (score={sc:.4f})  patient={pid} modality={modality}")

# --------- MAIN (interactive example) ---------
if __name__ == "__main__":
    query = input("Enter a query for retrieval (e.g. 'previous abdominal CT for pancreas'): ").strip()
    if not query:
        print("No query entered, exiting.")
        exit(0)

    q_emb = encode_query(query)  # shape (1, dim)
    d_txt, i_txt = search_index(txt_index, q_emb, k=TOP_K)
    d_img, i_img = search_index(img_index, q_emb, k=TOP_K)

    txt_files = [txt_keys[idx].item() if isinstance(txt_keys[idx], np.ndarray) or isinstance(txt_keys[idx], np.generic) else txt_keys[idx] for idx in i_txt]
    img_files = [img_keys[idx].item() if isinstance(img_keys[idx], np.ndarray) or isinstance(img_keys[idx], np.generic) else img_keys[idx] for idx in i_img]

    pretty_print_results(txt_files, d_txt, "TEXT INDEX")
    pretty_print_results(img_files, d_img, "IMAGE INDEX")

    # Optionally: gather top-N text 'findings+impression' to feed into summarizer
    top_text_rows = meta[meta["filename"].isin(txt_files)]
    # order by score alignment
    print("\n--- Top text snippets (findings + impression) ---")
    for fname in txt_files:
        row = meta[meta["filename"] == fname]
        if not row.empty:
            print(f"\nFilename: {fname}")
            print("Findings:", row.iloc[0]["findings"])
            print("Impression:", row.iloc[0]["impression"])
