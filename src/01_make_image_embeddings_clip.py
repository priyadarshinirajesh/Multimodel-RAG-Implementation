# src/01_make_image_embeddings_clip.py
import os, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

META = "data/raw/final_multimodal_dataset.csv"
OUT_DIR = "outputs/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_EMB = os.path.join(OUT_DIR, "image_embeddings.npy")
OUT_KEYS = os.path.join(OUT_DIR, "image_keys.npy")

device = "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

df = pd.read_csv(META)
embs = []
keys = []

batch_size = 16  # lower on CPU if memory slow
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    imgs = []
    for _, row in batch.iterrows():
        p = row["filename"]
        if os.path.exists(p):
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except:
                imgs.append(Image.new("RGB",(224,224)))
        else:
            imgs.append(Image.new("RGB",(224,224)))
    inputs = processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).cpu().numpy()
    # normalize
    norms = (image_features**2).sum(axis=1, keepdims=True)**0.5
    image_features = image_features / (norms + 1e-10)
    embs.append(image_features)
    keys.extend(batch["filename"].tolist())

embs = np.vstack(embs).astype("float32")
np.save(OUT_EMB, embs)
np.save(OUT_KEYS, np.array(keys))
print("Saved image embeddings:", OUT_EMB)


