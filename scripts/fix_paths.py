import pandas as pd
import os

CSV_PATH = "data/raw/final_multimodal_dataset.csv"

df = pd.read_csv(CSV_PATH)

def fix_path(p):
    if isinstance(p, float) or not isinstance(p, str):
        return p

    # Convert Windows slashes → UNIX slashes
    p = p.replace("\\", "/")

    # Only keep part after 'dataset_indiana/'
    if "dataset_indiana/" in p:
        p = p.split("dataset_indiana/")[1]
        return "data/raw/dataset_indiana/" + p
    
    return p

df["filename"] = df["filename"].apply(fix_path)

df.to_csv("data/raw/final_multimodal_dataset_FIXED.csv", index=False)

print("✔ Paths fixed. Saved to final_multimodal_dataset_FIXED.csv")
