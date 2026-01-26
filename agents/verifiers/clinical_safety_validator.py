# agents/verifiers/clinical_safety_validator.py

import re
from utils.logger import get_logger

logger = get_logger("ClinicalSafetyValidator")

OVERCONFIDENT = [
    "definitely", "confirmed diagnosis", "100% certain", "patient has"
]

MEDICAL_ORDERS = [
    "start antibiotics", "perform thoracentesis",
    "admit patient immediately"
]

HEDGING_TERMS = [
    "suggests", "may indicate",
    "findings are consistent with",
    "clinical correlation recommended"
]


def validate_clinical_safety(response: str) -> dict:
    text = response.lower()
    issues = []

    if any(term in text for term in OVERCONFIDENT):
        issues.append("Overconfident diagnostic language")

    if any(order in text for order in MEDICAL_ORDERS):
        issues.append("Medical action/order detected")

    if not any(hedge in text for hedge in HEDGING_TERMS):
        issues.append("Missing clinical hedging language")

    if issues:
        logger.warning(f"[SafetyValidator] Issues detected: {issues}")
        return {"passed": False, "issues": issues}

    return {"passed": True, "issues": []}
