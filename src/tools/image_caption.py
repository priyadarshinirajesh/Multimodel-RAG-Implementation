# src/tools/image_caption.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Lazy loaded CLIP model
_clip_processor = None
_clip_model = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("Loading CLIP model for captioning...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def clip_caption(image_path):
    """Generate a caption using CLIP similarity ranking."""

    model, processor = _load_clip()

    image = Image.open(image_path).convert("RGB")

    # Candidate prompts
    prompts = [
        "A medical scan.",
        "An X-ray image.",
        "An MRI scan.",
        "A CT scan.",
        "An ultrasound scan.",
        "A radiology image showing abnormalities.",
        "A healthy organ scan.",
        "A scan showing disease."
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    best = torch.argmax(logits).item()

    return prompts[best]
