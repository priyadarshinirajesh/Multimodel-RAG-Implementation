# agents/verifiers/xray_retrieval_contract.py

from utils.logger import get_logger

logger = get_logger("XrayRetrievalContract")

def verify_xray_retrieval(evidence):
    issues = []
    warnings = []

    # ✅ Require at least 1 X-ray (not 2)
    if len(evidence) < 1:
        issues.append("No XRAY records retrieved")
    elif len(evidence) == 1:
        warnings.append("Only 1 XRAY view available")

    modalities = set()
    has_image = False
    relevance_scores = []

    for e in evidence:
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
        warnings.append("Low relevance scores (all < 0.3)")

    # ✅ Pass if we have valid X-rays
    passed = len(issues) == 0

    if not passed:
        logger.warning(f"[XrayRetrievalContract] FAILED - Issues: {issues}")
    elif warnings:
        logger.info(f"[XrayRetrievalContract] PASSED with warnings: {warnings}")
    else:
        logger.info(f"[XrayRetrievalContract] PASSED - {len(evidence)} X-rays retrieved")

    return {
        "passed": passed,
        "issues": issues + warnings  # ✅ Include warnings in issues list
    }