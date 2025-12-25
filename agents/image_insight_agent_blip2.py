# agents/image_insight_agent_blip2.py

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

MODEL_ID = "Salesforce/blip2-flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] Loading BLIP-2 model...")

processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

def image_insight_agent_blip2(evidence):
    insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [BLIP-2] Analyzing image: {e['image_path']}")
        image = Image.open(e["image_path"]).convert("RGB")

        prompt = (
            "Describe the visible medical findings in this image. "
            "Do not diagnose. Be factual."
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200
            )

        response = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        insights.append(f"[R{idx}-IMAGE] {response}")

    return insights
