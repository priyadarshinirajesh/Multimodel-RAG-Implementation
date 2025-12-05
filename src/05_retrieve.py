# src/05_retrieve.py

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel

from rbac import apply_rbac   # <-- IMPORT RBAC HERE

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
META = "data/raw/final_multimodal_dataset.csv"

IMG_INDEX = "outputs/embeddings/faiss_image.index"
TXT_INDEX = "outputs/embeddings/faiss_text.index"

IMG_KEYS = "outputs/embeddings/image_keys.npy"
TXT_KEYS = "outputs/embeddings/text_keys.npy"

DEVICE = "cpu"
TOP_K = 5

# ------------------------------------------------------------
# LOAD DATA AND MODELS
# ------------------------------------------------------------
meta = pd.read_csv(META)

# MPNet (for TEXT)
mpnet_name = "sentence-transformers/all-mpnet-base-v2"
tok_mpnet = AutoTokenizer.from_pretrained(mpnet_name)
model_mpnet = AutoModel.from_pretrained(mpnet_name).to(DEVICE)

# CLIP text encoder (for IMAGE queries)
clip_name = "openai/clip-vit-base-patch32"
tok_clip = CLIPTokenizer.from_pretrained(clip_name)
model_clip = CLIPModel.from_pretrained(clip_name).to(DEVICE)

# Load FAISS indexes
txt_index = faiss.read_index(TXT_INDEX)
img_index = faiss.read_index(IMG_INDEX)

txt_keys = np.load(TXT_KEYS, allow_pickle=True)
img_keys = np.load(IMG_KEYS, allow_pickle=True)

# ------------------------------------------------------------
# ENCODERS
# ------------------------------------------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def encode_text_mpnet(text):
    enc = tok_mpnet([text], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model_mpnet(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, dim=1)
    return pooled.cpu().numpy().astype("float32")

def encode_text_clip(text):
    enc = tok_clip([text], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model_clip.get_text_features(**enc)
    out = torch.nn.functional.normalize(out, dim=1)
    return out.cpu().numpy().astype("float32")

# ------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------
def search(index, vector, k=TOP_K):
    faiss.normalize_L2(vector)
    D, I = index.search(vector, k)
    return D[0], I[0]

# ------------------------------------------------------------
# PIPELINE: QUERY → RETRIEVE → RBAC → RETURN RESULTS
# ------------------------------------------------------------
def retrieve(query_text, role="doctor"):
    print(f"\n🔍 Running retrieval for: {query_text}\n")

    q_txt = encode_text_mpnet(query_text)
    q_img = encode_text_clip(query_text)

    # TEXT SEARCH
    d_t, i_t = search(txt_index, q_txt, TOP_K)
    text_files = [txt_keys[i] for i in i_t]

    # IMAGE SEARCH
    d_i, i_i = search(img_index, q_img, TOP_K)
    image_files = [img_keys[i] for i in i_i]

    # --------------------------------------------------------
    # Convert retrieved filenames → metadata rows
    # --------------------------------------------------------
    retrieved = []

    for f in text_files + image_files:
        row = meta[meta["filename"] == f]
        if row.empty:
            continue

        r = row.iloc[0]
        retrieved.append({
            "filename": r["filename"],
            "patient_id": r["patient_id"],
            "modality": r["modality"],
            "findings": r["findings"],
            "impression": r["impression"]
        })

    # --------------------------------------------------------
    # APPLY RBAC FILTERING
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


