# embeddings/image_embeddings.py

import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

# -----------------------------
# Device selection
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔍 Using device for image embeddings:", DEVICE)

# -----------------------------
# Load CLIP image model ON DEVICE
# -----------------------------
model = SentenceTransformer(
    "clip-ViT-B-32",
    device=DEVICE
)

print("🔍 Image model loaded on:", model.device)

# -----------------------------
# Image embedding function
# -----------------------------
def embed_image(image_path: str):
    """
    Input  : path to .png image
    Output : 512-dim image embedding (list[float])
    """
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        embedding = model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embedding.tolist()
