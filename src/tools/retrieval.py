# src/tools/retrieval.py

import os
import faiss
import numpy as np
import pandas as pd
import torch
# from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel # these are clip imports

from transformers import (
    AutoTokenizer,
    AutoModel,
    FlavaProcessor,
    FlavaModel
)

# ====================================================
# FIXED PROJECT ROOT PATH (VERY IMPORTANT)
# src/tools → src → project_root
# ====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Dataset location
META = os.path.join(PROJECT_ROOT, "data", "raw", "final_multimodal_dataset.csv")

# FAISS index + key paths
IMG_INDEX_PATH = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "faiss_image_flava.index")
TXT_INDEX_PATH = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "faiss_text.index")
IMG_KEYS = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "image_keys_flava.npy")
TXT_KEYS = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "text_keys.npy")

# Model names
MPNET = "sentence-transformers/all-mpnet-base-v2"
CLIP = "openai/clip-vit-base-patch32"

DEVICE = "cpu"

meta = pd.read_csv(META)

# -------------------------------
# HELPERS
# -------------------------------
def normalize_path(path):
    """Ensure Streamlit can load the image by normalizing file paths."""
    if isinstance(path, str):
        path = path.replace("\\", "/")
    return os.path.abspath(path)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


# -------------------------------
# LAZY-LOAD MODELS
# -------------------------------
_mpnet_model = None
_mpnet_tokenizer = None

# _clip_model = None
# _clip_tokenizer = None

_txt_index = None
_img_index = None
_txt_keys = None
_img_keys = None


# -------------------------------
# FLAVA (TEXT → IMAGE SEARCH)
# -------------------------------
_flava_model = None
_flava_processor = None

def _load_flava():
    global _flava_model, _flava_processor
    if _flava_model is None:
        print("Loading FLAVA model...")
        _flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        _flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(DEVICE)
    return _flava_processor, _flava_model


def _load_text_encoder():
    global _mpnet_model, _mpnet_tokenizer
    if _mpnet_model is None:
        print("Loading MPNet text encoder...")
        _mpnet_tokenizer = AutoTokenizer.from_pretrained(MPNET)
        _mpnet_model = AutoModel.from_pretrained(MPNET).to(DEVICE)
    return _mpnet_tokenizer, _mpnet_model


# def _load_clip_encoder():
#     global _clip_tokenizer, _clip_model
#     if _clip_model is None:
#         print("Loading CLIP encoder...")
#         _clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP)
#         _clip_model = CLIPModel.from_pretrained(CLIP).to(DEVICE)
#     return _clip_tokenizer, _clip_model


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

    print("[DEBUG] Image index dim:", _img_index.d)
    print("[DEBUG] Image keys count:", len(_img_keys))

    return _txt_index, _img_index, _txt_keys, _img_keys


# -------------------------------
# ENCODING FUNCTIONS
# -------------------------------
def encode_text_query(query):
    tok, model = _load_text_encoder()

    enc = tok([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        output = model(**enc)

    pooled = mean_pooling(output, enc["attention_mask"])
    emb = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return emb.cpu().numpy().astype("float32")


# def encode_clip_text(query):
#     tok, model = _load_clip_encoder()

#     enc = tok([query], padding=True, truncation=True, max_length=77, return_tensors="pt")
#     enc = {k: v.to(DEVICE) for k, v in enc.items()}

#     with torch.no_grad():
#         output = model.get_text_features(**enc)

#     emb = torch.nn.functional.normalize(output, p=2, dim=1)

#     return emb.cpu().numpy().astype("float32")

def encode_flava_text(query):
    """
    Encode text using FLAVA text encoder
    Output shape: (1, 768)
    """
    processor, model = _load_flava()

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        # outputs = model.get_text_features(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"]
        # )

    # emb = torch.nn.functional.normalize(outputs, p=2, dim=1)
    # return emb.cpu().numpy().astype("float32")

    pooled = outputs.mean(dim=1)  # (1, 768)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    print("[DEBUG] FLAVA pooled query shape:", pooled.shape)
    return pooled.cpu().numpy().astype("float32")


# -------------------------------
# FAISS SEARCH
# -------------------------------
def _search(index, q_emb, k):
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return D[0], I[0]


# -------------------------------
# PUBLIC RETRIEVAL FUNCTIONS
# -------------------------------
def retrieve_text(query, k=5):
    txt_idx, _, txt_keys, _ = _load_indexes()

    q_emb = encode_text_query(query)

    # dimension check
    if q_emb.shape[1] != txt_idx.d:
        raise ValueError(f"Dimension mismatch: query dim {q_emb.shape[1]} vs index dim {txt_idx.d}")

    D, I = _search(txt_idx, q_emb, k)

    results = []

    for score, idx in zip(D, I):
        fname = txt_keys[idx]

        row = meta[meta["filename"] == fname]
        if row.empty:
            continue

        results.append({
            "filename": normalize_path(fname),
            "patient_id": int(row["patient_id"].values[0]),
            "modality": row["modality"].values[0],
            "findings": row.iloc[0].get("findings", ""),
            "impression": row.iloc[0].get("impression", ""),
            "score": float(score)
        })

    return results


def retrieve_image(query, k=5):
    _, img_idx, _, img_keys = _load_indexes()

    #q_emb = encode_clip_text(query)
    q_emb = encode_flava_text(query)

    print("[DEBUG] FLAVA image-query shape:", q_emb.shape)

    if q_emb.shape[1] != img_idx.d:
        raise ValueError(f"Query emb dim {q_emb.shape[1]} != index dim {img_idx.d}")

    D, I = _search(img_idx, q_emb, k)

    results = []

    for score, idx in zip(D, I):
        fname = img_keys[idx].item()

        row = meta[meta["filename"] == fname]
        if row.empty:
            continue

        results.append({
            "filename": normalize_path(fname),
            "patient_id": int(row["patient_id"].values[0]),
            "modality": row["modality"].values[0],
            "findings": row.iloc[0].get("findings", ""),
            "impression": row.iloc[0].get("impression", ""),
            "score": float(score)
        })

    return results