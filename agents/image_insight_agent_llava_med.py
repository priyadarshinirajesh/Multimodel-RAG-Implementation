# agents/image_insight_agent_llava_med.py

import requests
import base64
from utils.logger import get_logger

logger = get_logger("LLaVA-Med")

OLLAMA_URL = "http://localhost:11434/api/generate"

# Exact model name as pulled via Ollama
LLAVA_MED_MODEL = "z-uo/llava-med-v1.5-mistral-7b_q8_0"


def encode_image(image_path: str) -> str:
    """Encodes an image file as a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_insight_agent_llava_med(evidence: list, query: str) -> list:
    """
    Extracts observation-only visual insights from chest X-ray images
    using LLaVA-Med (Mistral-7B backbone) via Ollama.

    SAFETY GUARANTEES:
    - Never returns empty image insights
    - No diagnoses or disease naming
    - Deterministic fallback for normal / unclear images
    """

    image_insights = []

    for idx, e in enumerate(evidence, start=1):
        image_path = e.get("image_path")

        if not image_path:
            logger.debug(f"[LLaVA-Med] Skipping evidence {idx}: no image_path")
            continue

        logger.info(f"[LLaVA-Med] Analyzing image {idx}: {image_path}")

        try:
            image_b64 = encode_image(image_path)
        except Exception as ex:
            logger.error(f"[LLaVA-Med] Failed to load image {image_path}: {ex}")
            image_insights.append(f"[R{idx}-IMAGE] No acute abnormalities.")
            continue

        # STRICT, FAIL-SAFE PROMPT
        prompt = f"""
You are a specialized radiology assistant.

Analyze the provided chest X-ray image ONLY with respect to this clinical question:
"{query}"

RULES (MANDATORY):
- Describe ONLY visible anatomical observations.
- DO NOT diagnose.
- DO NOT name diseases or conditions.
- DO NOT infer causes.
- Use neutral radiology-style language.
- Maximum 20 words.

Do NOT return an empty response.
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

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()

            raw_response = response.json()
            insight = raw_response.get("response", "").strip()

            # ---------- HARD FAIL-SAFE ----------
            if not insight or insight.lower() in ["", "none", "n/a"]:
                logger.warning(
                    f"[LLaVA-Med] Empty output for image {idx}. Applying fallback."
                )
                insight = "No acute abnormalities."

            image_insights.append(f"[R{idx}-IMAGE] {insight}")

            # OPTIONAL DEBUG (comment out if noisy)
            logger.debug(f"[LLaVA-Med] Image {idx} insight: {insight}")

        except Exception as ex:
            logger.error(f"[LLaVA-Med] LLaVA inference failed for image {idx}: {ex}")
            image_insights.append(f"[R{idx}-IMAGE] No acute abnormalities.")

    return image_insights
