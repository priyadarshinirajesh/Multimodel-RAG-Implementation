# agents/image_insight_agent.py

import requests
import base64
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_insight_agent(evidence):
    """
    Uses LLaVA-Med via Ollama to extract visual findings from medical images
    """

    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        image_path = e.get("image_path")

        if not image_path or not os.path.exists(image_path):
            continue

        print(f"[INFO] [ImageInsightAgent] Analyzing image: {image_path}")

        image_base64 = encode_image_base64(image_path)

        prompt = f"""
You are a medical vision-language model.

Analyze the medical image and describe:
- Key anatomical observations
- Any abnormal visual findings
- Relevant diagnostic clues

Be factual. Do not guess.

Imaging modality: {e.get('modality')}
"""

        payload = {
            "model": "llava-med",
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)

        if response.status_code != 200:
            print(
                f"[ERROR] [ImageInsightAgent] Ollama failed for {image_path} | "
                f"Status: {response.status_code}"
            )
            continue

        insight = response.json().get("response", "")

        print(f"[DEBUG] [ImageInsightAgent] Insight R{idx}: {insight}")

        image_insights.append(
            f"[R{idx}-IMAGE] {insight}"
        )

    return image_insights
