# agents/image_insight_agent_ollama.py

import requests
import base64
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_insight_agent_ollama(evidence,query:str):
    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [LLaVA] Analyzing image: {e['image_path']}")

        image_b64 = encode_image(e["image_path"])

        prompt = """
You are a medical image observation assistant.

CLINICAL QUESTION:
"{user_query}"

TASK:
Describe ONLY visual findings that are RELEVANT to the clinical question.

STRICT RULES (MANDATORY):
- DO NOT name diseases or diagnoses.
- DO NOT make clinical conclusions.
- DO NOT speculate or infer beyond what is visible.
- Describe anatomy ONLY if it is clearly visible AND relevant.
- If the image does NOT show information relevant to the question,
  write EXACTLY: "Unclear from image."
- If the image is unrelated to the clinical question,
  write EXACTLY: "Not relevant to the query."

OUTPUT CONSTRAINTS:
- Maximum 2 short sentences.
- Each sentence ≤ 12 words.
- Use neutral, observational language only.
- NO extra explanations.

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

