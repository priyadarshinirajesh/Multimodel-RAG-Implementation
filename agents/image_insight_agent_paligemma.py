# agents/image_insight_agent_paligemma.py

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from utils.logger import get_logger

logger = get_logger("ImageInsightAgent")

MODEL_ID = "google/paligemma-3b-mix-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

logger.info("Loading PaliGemma 3B model...")

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)


def image_insight_agent_paligemma(evidence):
    """
    Extracts medical image observations using PaliGemma.
    Returns image-derived textual evidence.
    """

    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        image_path = e["image_path"]

        try:
            logger.info(f"Analyzing image: {image_path}")

            image = Image.open(image_path).convert("RGB")

            prompt = (
                "Describe visible anatomical structures and abnormalities "
                "in this medical image. Be factual. Do not speculate."
            )

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )

            generated = output[0][input_len:]
            response = processor.decode(
                generated, skip_special_tokens=True
            )

            logger.debug(f"PaliGemma Insight R{idx}: {response}")

            image_insights.append(f"[R{idx}-IMAGE] {response}")

        except Exception as ex:
            logger.error(f"PaliGemma failed on {image_path}: {ex}")

    return image_insights
