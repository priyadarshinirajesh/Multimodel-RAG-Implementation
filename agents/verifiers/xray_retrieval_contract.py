# agents/verifiers/xray_retrieval_contract.py

from utils.logger import get_logger

logger = get_logger("XrayRetrievalContract")

def verify_xray_retrieval(evidence):
    issues = []

    if len(evidence) < 2:
        issues.append("Less than 2 XRAY records retrieved")

    modalities = set()
    has_image = False
    relevance_scores = []

    for e in evidence:
        # Handle Qdrant ScoredPoint
        payload = e.payload if hasattr(e, "payload") else e

        modality = payload.get("modality")
        if modality:
            modalities.add(modality)

        if payload.get("image_path"):
            has_image = True

        if hasattr(e, "score"):
            relevance_scores.append(e.score)

    if modalities != {"XRAY"}:
        issues.append("Non-XRAY modality detected")

    if not has_image:
        issues.append("No X-ray image found in retrieved evidence")

    if not any(score > 0.3 for score in relevance_scores):
        issues.append("No evidence with relevance score > 0.3")

    passed = len(issues) == 0

    if not passed:
        logger.warning(f"[XrayRetrievalContract] Issues: {issues}")

    return {
        "passed": passed,
        "issues": issues
    }
