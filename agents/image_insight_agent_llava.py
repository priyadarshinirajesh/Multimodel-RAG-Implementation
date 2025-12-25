# agents/image_insight_agent_llava.py

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] Loading LLaVA model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

def image_insight_agent_llava(evidence):
    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [LLaVA] Analyzing image: {e['image_path']}")

        image = Image.open(e["image_path"]).convert("RGB")

        prompt = (
            "USER: Describe visible medical findings in this image. "
            "Only describe what is visually observable. Do not diagnose.\n"
            "<image>\n"
            "ASSISTANT:"
        )

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        response = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        image_insights.append(f"[R{idx}-IMAGE] {response}")

    return image_insights

