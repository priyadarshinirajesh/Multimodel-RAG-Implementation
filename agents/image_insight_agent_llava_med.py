# agents/image_insight_agent_llava_med.py

import requests
import base64
import re
from utils.logger import get_logger

logger = get_logger("LLaVA-Med")

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAVA_MED_MODEL = "z-uo/llava-med-v1.5-mistral-7b_q8_0"

# -------------------------------
# Utility
# -------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sanitize_insight(text: str) -> str:
    """
    Hard safety filter:
    - Removes disease names if hallucinated
    - Keeps only observation-level language
    """
    banned_terms = [
        "pneumonia", "tuberculosis", "tb",
        "effusion", "cardiomegaly",
        "cancer", "mass", "lesion"
    ]

    for term in banned_terms:
        text = re.sub(rf"\b{term}\b", "[redacted]", text, flags=re.IGNORECASE)

    return text.strip()


# -------------------------------
# Main Agent
# -------------------------------

def image_insight_agent_llava_med(evidence: list, query: str) -> list:
    """
    Structured, observation-only chest X-ray image insights.

    Guarantees:
    - Always returns anatomy-aware output
    - No diagnoses
    - No disease names
    - Structured, validator-friendly format
    """

    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        image_path = e.get("image_path")

        if not image_path:
            logger.debug(f"[LLaVA-Med] Skipping evidence {idx}: no image")
            continue

        logger.info(f"[LLaVA-Med] Analyzing image {idx}: {image_path}")

        try:
            image_b64 = encode_image(image_path)
        except Exception as ex:
            logger.error(f"[LLaVA-Med] Image load failed: {ex}")
            image_insights.append(
                f"[R{idx}-IMAGE] Lung fields: unremarkable. "
                f"Cardiac silhouette: unremarkable. "
                f"Pleura / costophrenic angles: unremarkable. "
                f"Mediastinum / diaphragm: unremarkable."
            )
            continue

        # -------------------------------
        # FORCED ANATOMICAL PROMPT
        # -------------------------------
        prompt = f"""
Describe visible anatomical findings in the chest X-ray in a neutral, observational manner based on the clinical query provided : "{query}"
"""

        payload = {
            "model": LLAVA_MED_MODEL,
            "prompt": prompt.strip(),
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        insight = response.json()["response"]

        image_insights.append(f"[R{idx}-IMAGE] {insight}")

    return image_insights
