import os
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel

# ========= CONFIG ==========
META = "data/raw/final_multimodal_dataset.csv"

IMG_INDEX_PATH = "outputs/faiss/faiss_image.index"     # dimension = 512
TXT_INDEX_PATH = "outputs/faiss/faiss_text.index"      # dimension = 768

IMG_KEYS = "outputs/embeddings/image_keys.npy"
TXT_KEYS = "outputs/embeddings/text_keys.npy"

DEVICE = "cpu"
TOP_K = 5


# ========= LOAD MODELS ==========
print("\nLoading MPNet (for TEXT retrieval)...")
MPNET = "sentence-transformers/all-mpnet-base-v2"
mpnet_tok = AutoTokenizer.from_pretrained(MPNET)
mpnet_model = AutoModel.from_pretrained(MPNET).to(DEVICE)

print("Loading CLIP text encoder (for IMAGE retrieval)...")
CLIP = "openai/clip-vit-base-patch32"
clip_tok = CLIPTokenizer.from_pretrained(CLIP)
clip_model = CLIPModel.from_pretrained(CLIP).to(DEVICE)


# ========= LOAD FAISS ==========
print("\nLoading FAISS indexes...")
txt_index = faiss.read_index(TXT_INDEX_PATH)  # 768-d
img_index = faiss.read_index(IMG_INDEX_PATH)  # 512-d

txt_keys = np.load(TXT_KEYS, allow_pickle=True)
img_keys = np.load(IMG_KEYS, allow_pickle=True)

meta = pd.read_csv(META)


# ========= UTILITIES ==========
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_mpnet(query):
    """Encode with MPNet → 768-d"""
    enc = mpnet_tok([query], padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = mpnet_model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype("float32")


def encode_clip(query):
    """Encode with CLIP text encoder → 512-d"""
    enc = clip_tok([query], padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        z = clip_model.get_text_features(**enc)
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return z.cpu().numpy().astype("float32")


def search(index, q_emb, k):
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0], I[0]


def pretty_print(files, scores, label):
    print(f"\nTop {len(files)} results from {label}:")
    for fname, s in zip(files, scores):
        row = meta[meta["filename"] == fname]
        if not row.empty:
            mod = row.iloc[0]["modality"]
            pid = row.iloc[0]["patient_id"]
        else:
            mod = pid = "N/A"
        print(f"- {fname} | score={s:.4f} | modality={mod} | patient={pid}")


# ========= MAIN ==========
if __name__ == "__main__":

    query = input("\nEnter query: ").strip()
    if not query:
        print("Empty query, exiting.")
        exit()

    print("\nEncoding query with MPNet (TEXT search)...")
    q_txt = encode_mpnet(query)

    print("Encoding query with CLIP (IMAGE search)...")
    q_img = encode_clip(query)

    # ----- FAISS search -----
    d_txt, i_txt = search(txt_index, q_txt, TOP_K)
    d_img, i_img = search(img_index, q_img, TOP_K)

    txt_results = [txt_keys[i] for i in i_txt]
    img_results = [img_keys[i] for i in i_img]

    pretty_print(txt_results, d_txt, "TEXT INDEX")
    pretty_print(img_results, d_img, "IMAGE INDEX")

    # ----- Print clinical text snippets -----
    print("\n--- Top retrieved text snippets ---")
    for f in txt_results:
        row = meta[meta["filename"] == f]
        if not row.empty:
            print(f"\n{f}")
            print("Findings:", row.iloc[0]["findings"])
            print("Impression:", row.iloc[0]["impression"])
