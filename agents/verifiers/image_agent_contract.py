# agents/verifiers/image_agent_contract.py

import re
from utils.logger import get_logger

logger = get_logger("ImageAgentContract")

BANNED_TERMS = [
    "pneumonia", "tb", "tuberculosis", "cancer", "lesion", "mass"
]

ALLOWED_ANATOMY = [
    "lung", "lungs", "cardiac", "heart",
    "pleura", "diaphragm", "mediastinum"
]


def verify_image_insights(image_insights: list) -> dict:
    issues = []

    if not image_insights:
        return {"passed": False, "issues": ["No image insights returned"]}

    for insight in image_insights:
        text = insight.lower()

        if len(text.split()) < 5:
            issues.append("Image insight too short")

        if any(term in text for term in BANNED_TERMS):
            issues.append("Diagnosis term detected in image insight")

        if not any(anat in text for anat in ALLOWED_ANATOMY):
            issues.append("No chest anatomy mentioned")

    if issues:
        logger.warning(f"[ImageContract] Issues detected: {issues}")
        return {"passed": False, "issues": issues}

    return {"passed": True, "issues": []}
