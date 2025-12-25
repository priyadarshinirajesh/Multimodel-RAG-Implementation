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
You are a medical image observation assistant.

TASK:
Describe ONLY what is directly visible in this image.

STRICT RULES:
- DO NOT name diseases or diagnoses.
- DO NOT guess organ unless visually obvious.
- Use anatomical terms ONLY if clearly visible.
- If a structure is unclear, write exactly: "Unclear from image."
- Maximum 3 short sentences.
- Each sentence ≤ 12 words.
- NO speculation.
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

