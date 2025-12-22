# src/05_retrieve.py

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import FlavaProcessor, FlavaModel

from rbac import apply_rbac

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
META = "data/raw/final_multimodal_dataset.csv"

IMG_INDEX = "outputs/embeddings/faiss_image_flava.index"
TXT_INDEX = "outputs/embeddings/faiss_text_flava.index"

IMG_KEYS = "outputs/embeddings/image_keys_flava.npy"
TXT_KEYS = "outputs/embeddings/text_keys_flava.npy"

DEVICE = "cpu"
TOP_K = 5

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
meta = pd.read_csv(META)

# ------------------------------------------------------------
# LOAD FLAVA MODEL
# ------------------------------------------------------------
print("🔵 Loading FLAVA model...")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full").to(DEVICE)
model.eval()

# ------------------------------------------------------------
# LOAD FAISS INDEXES
# ------------------------------------------------------------
txt_index = faiss.read_index(TXT_INDEX)
img_index = faiss.read_index(IMG_INDEX)

txt_keys = np.load(TXT_KEYS, allow_pickle=True)
img_keys = np.load(IMG_KEYS, allow_pickle=True)

# ------------------------------------------------------------
# ENCODERS
# ------------------------------------------------------------
def encode_text_flava(text):
    inputs = processor(text=text, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # token-level → mean pooling
    token_embeds = outputs.text_embeddings.squeeze(0)
    emb = token_embeds.mean(dim=0)

    emb = torch.nn.functional.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy().astype("float32").reshape(1, -1)


# ------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------
def search(index, vector, k=TOP_K):
    faiss.normalize_L2(vector)
    D, I = index.search(vector, k)
    return D[0], I[0]


# ------------------------------------------------------------
# PIPELINE: QUERY → RETRIEVE → RBAC
# ------------------------------------------------------------
def retrieve(query_text, role="doctor"):
    print(f"\n🔍 Running FLAVA retrieval for: {query_text}\n")

    q_vec = encode_text_flava(query_text)

    # TEXT SEARCH
    d_t, i_t = search(txt_index, q_vec, TOP_K)
    text_files = [txt_keys[i] for i in i_t]

    # IMAGE SEARCH (same query vector)
    d_i, i_i = search(img_index, q_vec, TOP_K)
    image_files = [img_keys[i] for i in i_i]

    retrieved = []

    for f in list(dict.fromkeys(text_files + image_files)):
        row = meta[meta["filename"] == f]
        if row.empty:
            continue

        r = row.iloc[0]
        retrieved.append({
            "filename": r["filename"],
            "patient_id": r.get("patient_id"),
            "modality": r.get("modality"),
            "findings": r.get("findings", ""),
            "impression": r.get("impression", "")
        })

    # --------------------------------------------------------
    # APPLY RBAC
    # --------------------------------------------------------
    filtered = apply_rbac(role, retrieved)

    print(f"\n🔐 RBAC Applied (role = {role})")
    print(f"Returned {len(filtered)} items\n")

    for item in filtered:
        print(item)
        print("-" * 60)

    return filtered


# ------------------------------------------------------------
# MANUAL TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    query = input("Enter your medical query: ")
    role = input("Enter role (doctor/nurse/patient/admin): ")

    retrieve(query, role)
