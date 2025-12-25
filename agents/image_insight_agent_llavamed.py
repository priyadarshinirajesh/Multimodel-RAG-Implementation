# agents/image_insight_agent_llavamed.py

import requests
import base64
from utils.logger import get_logger

logger = get_logger("ImageInsightAgent")

LLAVA_CONTROLLER = "http://localhost:10000"
MODEL_NAME = "llava-med-v1.5-mistral-7b"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_insight_agent_llavamed(evidence):
    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        logger.info(f"Analyzing image via LLaVA-Med: {e['image_path']}")

        image_b64 = encode_image(e["image_path"])

        prompt = (
            "You are a biomedical vision-language model.\n"
            "Analyze the medical image and describe:\n"
            "- Key anatomical observations\n"
            "- Visible abnormalities\n"
            "- Clinically relevant findings\n"
            "Be factual. Do not speculate."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_b64}
                    ]
                }
            ]
        }

        response = requests.post(
            f"{LLAVA_CONTROLLER}/v1/chat/completions",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        insight = response.json()["choices"][0]["message"]["content"]

        logger.debug(f"LLaVA-Med Insight R{idx}: {insight}")
        image_insights.append(f"[R{idx}-IMAGE] {insight}")

    return image_insights

