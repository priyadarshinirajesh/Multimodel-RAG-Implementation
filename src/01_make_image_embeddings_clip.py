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
REPORT_FILE = os.path.join(OUT_DIR, "embedding_report.txt")

device = "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

df = pd.read_csv(META)
embs = []
keys = []

# Tracking counters
valid_images = 0
missing_images = 0
failed_to_load = 0

missing_list = []
failed_list = []

batch_size = 16

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    imgs = []
    
    for _, row in batch.iterrows():
        p = row["filename"]

        if os.path.exists(p):  
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_images += 1
            except Exception as e:
                imgs.append(Image.new("RGB",(224,224)))
                failed_images += 1
                failed_list.append(p)
        else:
            imgs.append(Image.new("RGB",(224,224)))
            missing_images += 1
            missing_list.append(p)
    
    inputs = processor(images=imgs, return_tensors="pt")
    
    with torch.no_grad():
        features = model.get_image_features(**inputs).cpu().numpy()

    norms = (features**2).sum(axis=1, keepdims=True)**0.5
    features = features / (norms + 1e-10)

    embs.append(features)
    keys.extend(batch["filename"].tolist())

# Stack embeddings
embs = np.vstack(embs).astype("float32")
np.save(OUT_EMB, embs)
np.save(OUT_KEYS, np.array(keys))

print("\n✔ Saved image embeddings:", OUT_EMB)

# ----------------------------------------------
# WRITE REPORT FILE
# ----------------------------------------------
with open(REPORT_FILE, "w") as f:
    f.write("IMAGE EMBEDDING SUMMARY\n")
    f.write("=======================\n")
    f.write(f"Total rows in CSV: {len(df)}\n")
    f.write(f"Valid images processed: {valid_images}\n")
    f.write(f"Missing images: {missing_images}\n")
    f.write(f"Failed to load (corrupt): {failed_to_load}\n\n")

    if missing_list:
        f.write("Missing files:\n")
        for x in missing_list:
            f.write(x + "\n")
    
    if failed_list:
        f.write("\nFailed to load:\n")
        for x in failed_list:
            f.write(x + "\n")

print("\n📌 Summary Report saved at:", REPORT_FILE)
print(f"➡ Valid images: {valid_images}")
print(f"➡ Missing images: {missing_images}")
print(f"➡ Failed images: {failed_to_load}")
print("Done.\n")
