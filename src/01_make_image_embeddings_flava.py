# src/01_make_image_embeddings_flava.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from transformers import CLIPProcessor, CLIPModel
from transformers import FlavaProcessor, FlavaModel
import torch

META = "data/raw/final_multimodal_dataset.csv"
OUT_DIR = "outputs/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EMB = os.path.join(OUT_DIR, "image_embeddings_flava.npy")
OUT_KEYS = os.path.join(OUT_DIR, "image_keys_flava.npy")
REPORT_FILE = os.path.join(OUT_DIR, "embedding_report_flava.txt")

device = "cpu"
# model_name = "openai/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(model_name).to(device)
# processor = CLIPProcessor.from_pretrained(model_name)

# -----------------------------------
# LOAD FLAVA
# -----------------------------------
print("🔄 Loading FLAVA image encoder...")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
model.eval()

df = pd.read_csv(META)

embs = []
keys = []

# Tracking counters
valid_images = 0
missing_images = 0
failed_images = 0

missing_list = []
failed_list = []

batch_size = 8

# -----------------------------------
# MAIN LOOP
# -----------------------------------
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i + batch_size]
    images = []

    for _, row in batch.iterrows():
        path = row["filename"]

        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_images += 1
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
                failed_images += 1
                failed_list.append(path)
        else:
            images.append(Image.new("RGB", (224, 224)))
            missing_images += 1
            missing_list.append(path)

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        # 🔥 CRITICAL FIX: pool patch embeddings
        feats = feats.mean(dim=1)  # (batch, 768)

    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    if i == 0:
        print("[DEBUG] FLAVA image embedding shape:", feats.shape)
        # MUST BE: (batch_size, 768)

    embs.append(feats.cpu().numpy())
    keys.extend(batch["filename"].tolist())

# -----------------------------------
# SAVE
# -----------------------------------
embs = np.vstack(embs).astype("float32")

np.save(OUT_EMB, embs)
np.save(OUT_KEYS, np.array(keys))

print("\n✅ Saved FLAVA image embeddings:", OUT_EMB)
print("✅ Saved FLAVA image keys:", OUT_KEYS)

# ----------------------------------------------
# WRITE REPORT FILE
# ----------------------------------------------
with open(REPORT_FILE, "w") as f:
    f.write("FLAVA IMAGE EMBEDDING SUMMARY\n")
    f.write("============================\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Valid images: {valid_images}\n")
    f.write(f"Missing images: {missing_images}\n")
    f.write(f"Failed images: {failed_images}\n\n")

    if missing_list:
        f.write("Missing files:\n")
        for x in missing_list:
            f.write(x + "\n")

    if failed_list:
        f.write("\nFailed files:\n")
        for x in failed_list:
            f.write(x + "\n")

print("\n📄 Report written to:", REPORT_FILE)
print("🎉 FLAVA image embedding generation COMPLETE.")
