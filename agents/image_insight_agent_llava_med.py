# agents/image_insight_agent_llava_med.py

import requests
import base64
import re
from utils.logger import get_logger

logger = get_logger("LLaVA-Med")

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
LLAVA_MED_MODEL = "z-uo/llava-med-v1.5-mistral-7b_q8_0"

# ── Set to True to skip Ollama entirely and use structured fallback ────────────
# Useful when GPU memory is occupied by training or another model.
SKIP_LLAVA_MED = False


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sanitize_insight(text: str) -> str:
    banned_terms = [
        "pneumonia", "tuberculosis", "tb",
        "effusion", "cardiomegaly",
        "cancer", "mass", "lesion"
    ]
    for term in banned_terms:
        text = re.sub(rf"\b{term}\b", "[redacted]", text, flags=re.IGNORECASE)
    return text.strip()


def _ollama_is_available() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any(LLAVA_MED_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def _structured_fallback(idx: int, reason: str) -> str:
    return (
        f"[R{idx}-IMAGE] Based on imaging only (limited confidence): "
        f"Image analysis unavailable ({reason}). "
        f"Cardiac silhouette: not assessed. "
        f"Lung fields: not assessed. "
        f"Pleura and costophrenic angles: not assessed. "
        f"Mediastinum: not assessed."
    )


def image_insight_agent_llava_med(evidence: list, query: str) -> list:
    image_insights = []

    if SKIP_LLAVA_MED:
        logger.info("[LLaVA-Med] SKIP_LLAVA_MED=True — skipping all image analysis")
        for idx, e in enumerate(evidence, start=1):
            if e.get("image_path"):
                image_insights.append(_structured_fallback(idx, "image analysis disabled"))
        return image_insights

    if not _ollama_is_available():
        logger.warning(
            "[LLaVA-Med] Ollama not available — using structured fallback. "
            "This is normal when GPU memory is occupied by another process."
        )
        for idx, e in enumerate(evidence, start=1):
            if e.get("image_path"):
                image_insights.append(_structured_fallback(idx, "Ollama not available"))
        return image_insights

    for idx, e in enumerate(evidence, start=1):
        image_path = e.get("image_path")
        if not image_path:
            continue

        logger.info(f"[LLaVA-Med] Analyzing image {idx}: {image_path}")

        try:
            image_b64 = encode_image(image_path)
        except Exception as ex:
            logger.error(f"[LLaVA-Med] Image load failed for evidence {idx}: {ex}")
            image_insights.append(_structured_fallback(idx, f"image file error"))
            continue

        prompt = (
            f'Describe visible anatomical findings in the chest X-ray in a neutral, '
            f'observational manner relevant to: "{query}". '
            f'Focus on: lung fields, cardiac silhouette, pleura, costophrenic angles, '
            f'mediastinum, diaphragm. Do NOT provide a diagnosis.'
        )

        payload = {
            "model": LLAVA_MED_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            insight = sanitize_insight(response.json()["response"])
            image_insights.append(f"[R{idx}-IMAGE] {insight}")
            logger.info(f"[LLaVA-Med] Image {idx} analysis complete")

        except requests.exceptions.ConnectionError:
            logger.warning(f"[LLaVA-Med] Connection refused for image {idx}")
            image_insights.append(_structured_fallback(idx, "Ollama connection refused"))

        except requests.exceptions.Timeout:
            logger.warning(f"[LLaVA-Med] Timeout for image {idx}")
            image_insights.append(_structured_fallback(idx, "analysis timed out"))

        except Exception as ex:
            err_str = str(ex)
            if "500" in err_str or "memory" in err_str.lower():
                logger.warning(f"[LLaVA-Med] Ollama 500/OOM for image {idx} — insufficient GPU memory")
                image_insights.append(_structured_fallback(idx, "insufficient GPU memory"))
            else:
                logger.error(f"[LLaVA-Med] Unexpected error for image {idx}: {ex}")
                image_insights.append(_structured_fallback(idx, "unexpected error"))

    return image_insights