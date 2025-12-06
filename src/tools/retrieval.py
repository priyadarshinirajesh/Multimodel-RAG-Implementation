# src/tools/retrieval.py
import os
import time
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel

# CONFIG - adjust if your paths differ
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(ROOT, "data", "raw", "final_multimodal_dataset.csv")
IMG_INDEX_PATH = os.path.join(ROOT, "outputs", "embeddings", "faiss_image.index")
TXT_INDEX_PATH = os.path.join(ROOT, "outputs", "embeddings", "faiss_text.index")
IMG_KEYS = os.path.join(ROOT, "outputs", "embeddings", "image_keys.npy")
TXT_KEYS = os.path.join(ROOT, "outputs", "embeddings", "text_keys.npy")

# models
MPNET = "sentence-transformers/all-mpnet-base-v2"
CLIP = "openai/clip-vit-base-patch32"
DEVICE = "cpu"

# load meta
meta = pd.read_csv(META)

# pooling helper
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# load encoders (lazy init)
_mpnet_tokenizer = None
_mpnet_model = None
_clip_tokenizer = None
_clip_model = None
_txt_index = None
_img_index = None
_txt_keys = None
_img_keys = None

def _load_text_encoder():
    global _mpnet_model, _mpnet_tokenizer
    if _mpnet_model is None:
        print("Loading MPNet (text encoder)...")
        _mpnet_tokenizer = AutoTokenizer.from_pretrained(MPNET)
        _mpnet_model = AutoModel.from_pretrained(MPNET).to(DEVICE)
    return _mpnet_tokenizer, _mpnet_model

def _load_clip_encoder():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        print("Loading CLIP (text encoder for image retrieval)...")
        _clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP)
        _clip_model = CLIPModel.from_pretrained(CLIP).to(DEVICE)
    return _clip_tokenizer, _clip_model

def _load_indexes():
    global _txt_index, _img_index, _txt_keys, _img_keys
    if _txt_index is None:
        _txt_index = faiss.read_index(TXT_INDEX_PATH)
    if _img_index is None:
        _img_index = faiss.read_index(IMG_INDEX_PATH)
    if _txt_keys is None:
        _txt_keys = np.load(TXT_KEYS, allow_pickle=True)
    if _img_keys is None:
        _img_keys = np.load(IMG_KEYS, allow_pickle=True)
    return _txt_index, _img_index, _txt_keys, _img_keys

# encoders
def encode_text_query(query):
    tok, model = _load_text_encoder()
    enc = tok([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")

def encode_clip_text(query):
    tok, model = _load_clip_encoder()
    enc = tok([query], padding=True, truncation=True, max_length=77, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model.get_text_features(**enc)
    emb = torch.nn.functional.normalize(out, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")

# search helper
def _search(index, q_emb, k=5):
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0], I[0]

# public functions
def retrieve_text(query, k=5):
    txt_idx, _, txt_keys, _ = _load_indexes()
    q_emb = encode_text_query(query)
    # check dim
    if q_emb.shape[1] != txt_idx.d:
        raise ValueError(f"Dimension mismatch: query dim {q_emb.shape[1]} vs index dim {txt_idx.d}")
    D, I = _search(txt_idx, q_emb, k)
    results = []
    for score, idx in zip(D, I):
        fname = txt_keys[idx].item() if isinstance(txt_keys[idx], np.ndarray) or isinstance(txt_keys[idx], np.generic) else txt_keys[idx]
        row = meta[meta["filename"] == fname]
        if not row.empty:
            results.append({
                "filename": fname,
                "patient_id": int(row["patient_id"].values[0]) if "patient_id" in row.columns else None,
                "modality": row["modality"].values[0] if "modality" in row.columns else None,
                "findings": row.iloc[0].get("findings", ""),
                "impression": row.iloc[0].get("impression", ""),
                "score": float(score)
            })
    return results

def retrieve_image(query, k=5):
    _, img_idx, _, img_keys = _load_indexes()
    q_emb = encode_clip_text(query)
    if q_emb.shape[1] != img_idx.d:
        raise ValueError(f"Dimension mismatch: query dim {q_emb.shape[1]} vs index dim {img_idx.d}")
    D, I = _search(img_idx, q_emb, k)
    results = []
    for score, idx in zip(D, I):
        fname = img_keys[idx].item() if isinstance(img_keys[idx], np.ndarray) or isinstance(img_keys[idx], np.generic) else img_keys[idx]
        row = meta[meta["filename"] == fname]
        if not row.empty:
            results.append({
                "filename": fname,
                "patient_id": int(row["patient_id"].values[0]) if "patient_id" in row.columns else None,
                "modality": row["modality"].values[0] if "modality" in row.columns else None,
                "findings": row.iloc[0].get("findings", ""),
                "impression": row.iloc[0].get("impression", ""),
                "score": float(score)
            })
    return results
