# agents/verifiers/xray_anatomy_validator.py

from utils.logger import get_logger

logger = get_logger("XrayAnatomyValidator")

DISALLOWED_ANATOMY = [
    "pancreas", "liver", "kidney", "brain",
    "intestine", "gallbladder", "prostate"
]

DISALLOWED_MODALITIES = [
    "ct shows", "mri indicates", "ultrasound reveals"
]


def validate_xray_anatomy(response: str) -> dict:
    text = response.lower()
    issues = []

    for organ in DISALLOWED_ANATOMY:
        if organ in text:
            issues.append(f"Non-Xray anatomy mentioned: {organ}")

    for modality in DISALLOWED_MODALITIES:
        if modality in text:
            issues.append(f"Wrong modality mentioned: {modality}")

    if issues:
        logger.warning(f"[AnatomyValidator] Issues detected: {issues}")
        return {"passed": False, "issues": issues}

    return {"passed": True, "issues": []}
