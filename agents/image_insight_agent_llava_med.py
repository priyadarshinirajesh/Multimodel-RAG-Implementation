# agents/image_insight_agent_llava_med.py

import requests
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"

# Exact model name as pulled via Ollama
LLAVA_MED_MODEL = "z-uo/llava-med-v1.5-mistral-7b_q8_0"


def encode_image(image_path: str) -> str:
    """Encodes an image file as a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_insight_agent_llava_med(evidence, query: str):
    """
    Extracts observation-only visual insights from chest X-ray images
    using LLaVA-Med (Mistral-7B backbone) via Ollama.

    The agent is strictly constrained to avoid diagnosis or inference.
    """
    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        if not e.get("image_path"):
            continue

        print(f"[INFO] [LLaVA-Med] Analyzing image: {e['image_path']}")

        image_b64 = encode_image(e["image_path"])

        prompt = f"""
You are a medical image observation assistant trained on radiology images.

CLINICAL QUESTION:
"{query}"

TASK:
Describe ONLY observable visual findings in the image that are
RELEVANT to the clinical question.

STRICT RULES (MANDATORY):
- DO NOT name diseases, conditions, or diagnoses.
- DO NOT provide clinical interpretations or conclusions.
- DO NOT speculate beyond what is visibly present.
- Describe anatomy ONLY if it is clearly visible and relevant.
- If the image does NOT show information relevant to the question,
  respond EXACTLY with: "Unclear from image."
- If the image is unrelated to the clinical question,
  respond EXACTLY with: "Not relevant to the query."

OUTPUT CONSTRAINTS:
- Maximum 2 short sentences.
- Each sentence must be 12 words or fewer.
- Use neutral, radiology-style observational language only.
- NO explanations, NO bullet points, NO extra text.
"""

        payload = {
            "model": LLAVA_MED_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        insight = response.json().get("response", "").strip()

        image_insights.append(f"[R{idx}-IMAGE] {insight}")

    return image_insights
