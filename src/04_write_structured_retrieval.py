# src/04_write_structured_retrieval.py
import json, os, numpy as np, pandas as pd, faiss
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel
import torch

# Paths (adjust if you used other names)
META = "data/raw/final_multimodal_dataset.csv"
TXT_INDEX = "outputs/faiss/faiss_text.index"
IMG_INDEX = "outputs/faiss/faiss_image.index"
TXT_KEYS = "outputs/embeddings/text_keys.npy"
IMG_KEYS = "outputs/embeddings/image_keys.npy"

OUT_DIR = "outputs/retrieval"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "retrieved.json")

# Models (same as your retrieval script)
MPNET = "sentence-transformers/all-mpnet-base-v2"
CLIP = "openai/clip-vit-base-patch32"
DEVICE = "cpu"
TOP_K = 5

# Load meta & indexes
meta = pd.read_csv(META)
txt_index = faiss.read_index(TXT_INDEX)
img_index = faiss.read_index(IMG_INDEX)
txt_keys = np.load(TXT_KEYS, allow_pickle=True)
img_keys = np.load(IMG_KEYS, allow_pickle=True)

# Load encoders
mpnet_tok = AutoTokenizer.from_pretrained(MPNET)
mpnet_model = AutoModel.from_pretrained(MPNET).to(DEVICE)
clip_tok = CLIPTokenizer.from_pretrained(CLIP)
clip_model = CLIPModel.from_pretrained(CLIP).to(DEVICE)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def encode_mpnet(text):
    enc = mpnet_tok([text], padding=True, truncation=True, return_tensors="pt")
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        out = mpnet_model(**enc)
    pooled = mean_pooling(out, enc["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype("float32")

def encode_clip_text(text):
    enc = clip_tok([text], padding=True, truncation=True, return_tensors="pt")
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        out = clip_model.get_text_features(**enc)
    out = torch.nn.functional.normalize(out, p=2, dim=1)
    return out.cpu().numpy().astype("float32")

# Interactive query -> do both searches -> build structured retrieved list
query = input("Enter clinical query: ").strip()
if not query:
    print("No query entered. Exiting.")
    raise SystemExit(0)

q_txt = encode_mpnet(query)
q_img = encode_clip_text(query)

# search
faiss.normalize_L2(q_txt); faiss.normalize_L2(q_img)
D_txt, I_txt = txt_index.search(q_txt, TOP_K)
D_img, I_img = img_index.search(q_img, TOP_K)

items = []
# prefer text results for rich clinical text; but we include image hits as separate items (avoid duplicates)
for score, idx in zip(D_txt[0], I_txt[0]):
    fname = txt_keys[idx].item() if isinstance(txt_keys[idx], np.generic) else txt_keys[idx]
    row = meta[meta["filename"] == fname]
    if row.empty:
        continue
    r = row.iloc[0].to_dict()
    item = {
        "type": "text",
        "filename": fname,
        "patient_id": r.get("patient_id"),
        "modality": r.get("modality"),
        "findings": r.get("findings",""),
        "impression": r.get("impression",""),
        "score": float(score)
    }
    items.append(item)

# add image hits (if they are not already present)
present_fnames = {it["filename"] for it in items}
for score, idx in zip(D_img[0], I_img[0]):
    fname = img_keys[idx].item() if isinstance(img_keys[idx], np.generic) else img_keys[idx]
    if fname in present_fnames:
        continue
    row = meta[meta["filename"] == fname]
    r = row.iloc[0].to_dict() if not row.empty else {}
    item = {
        "type": "image",
        "filename": fname,
        "patient_id": r.get("patient_id"),
        "modality": r.get("modality"),
        "findings": r.get("findings",""),
        "impression": r.get("impression",""),
        "score": float(score)
    }
    items.append(item)

# save
with open(OUT_FILE, "w", encoding="utf-8") as fo:
    json.dump(items, fo, indent=2, ensure_ascii=False)

print(f"Saved {len(items)} retrieved items → {OUT_FILE}")
