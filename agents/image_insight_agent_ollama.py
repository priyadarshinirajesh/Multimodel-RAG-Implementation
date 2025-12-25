# agents/image_insight_agent_ollama.py

import requests
import base64
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_insight_agent_ollama(evidence):
    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [LLaVA] Analyzing image: {e['image_path']}")

        image_b64 = encode_image(e["image_path"])

        prompt = """
You are a medical imaging assistant.

Instructions:
- Describe ONLY visible anatomical structures
- Mention observable abnormalities (size, shape, opacity, shadow)
- Do NOT diagnose
- Do NOT speculate causes
- Be concise and factual
"""

        payload = {
            "model": "llava:7b",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        insight = response.json()["response"]

        image_insights.append(f"[R{idx}-IMAGE] {insight}")

    return image_insights

