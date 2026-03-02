# agents/verifiers/clinical_safety_validator.py

import re
from utils.logger import get_logger

logger = get_logger("ClinicalSafetyValidator")

# ── Overconfident language ─────────────────────────────────────────────────────
OVERCONFIDENT = [
    "definitely",
    "confirmed diagnosis",
    "100% certain",
    "patient has",
    "this is definitely",
    "without a doubt",
]

# ── Direct medical orders (AI should not issue these) ─────────────────────────
MEDICAL_ORDERS = [
    "start antibiotics",
    "perform thoracentesis",
    "admit patient immediately",
    "prescribe",
    "administer",
]

# ── Required hedging terms ─────────────────────────────────────────────────────
HEDGING_TERMS = [
    "suggests",
    "may indicate",
    "findings are consistent with",
    "clinical correlation",
    "cannot be excluded",
    "indeterminate",
    "insufficient evidence",
    "limited confidence",
    "high confidence",
    "moderate confidence",
    "low confidence",
]

# ── Filler recommendations that are useless in a clinical support tool ─────────
FILLER_RECOMMENDATIONS = [
    "further evaluation by a healthcare professional is necessary",
    "further evaluation by a healthcare professional",
    "consult a doctor",
    "seek medical advice",
    "see a doctor",
    "medical attention is advised",
]

# ── Patterns for specific, actionable recommendations ─────────────────────────
ACTIONABLE_PATTERNS = [
    r"(x-ray|chest\s*x.ray|ct|mri|ecg|ekg|echo|echocardiogram|ultrasound|pa\s*view|lateral\s*view)\s*(recommended|indicated|suggested|obtained|ordered)",
    r"(repeat|follow.up|compare\s*with\s*prior|interval\s*change)",
    r"(refer\s*to|specialist|cardiology|pulmonology|radiology|oncology)",
    r"\d+\s*(hour|day|week|month)",
    r"(lateral|pa\s*view|portable|upright)",
    r"(blood\s*pressure|o2|oxygen|saturation|spo2|ecg|ekg)",
    r"(biopsy|bronchoscopy|thoracentesis|pleurocentesis)",
]


def _extract_pathology_names_from_findings(pathology_findings: list) -> list:
    """
    Parse pathology finding strings like '[R1] - Effusion: 67.2% confidence'
    and return a list of lowercase pathology names.
    """
    names = []
    for finding in pathology_findings:
        # Match patterns like "- Effusion: 67.2%" or "Effusion: 67.2%"
        matches = re.findall(r'-\s*([A-Za-z][A-Za-z\s_]+):\s*\d+', finding)
        for m in matches:
            names.append(m.strip().lower())
    return names


def validate_clinical_safety(
    response: str,
    pathology_findings: list = None,
    evidence: list = None,
) -> dict:
    """
    Multi-layer clinical safety validation.

    Checks:
      1. Overconfident diagnostic language
      2. Direct medical orders
      3. Missing clinical hedging language
      4. Filler recommendations (Phase 2)
      5. Non-actionable recommendations (Phase 2)
      6. CNN finding omission — if DenseNet flagged something but response ignores it (Phase 2)

    Args:
        response:            The final generated clinical response string.
        pathology_findings:  Optional list of formatted pathology finding strings
                             from DenseNet (e.g. ["[R1] - Effusion: 67.2% confidence"]).
                             When provided, enables omission detection.

    Returns:
        dict with keys:
            passed  (bool)  — True if no issues found
            issues  (list)  — List of issue description strings
    """

    text = response.lower()
    issues = []

    # ── 1. Overconfident language ──────────────────────────────────────────────
    for term in OVERCONFIDENT:
        if term in text:
            issues.append(f"Overconfident diagnostic language: '{term}'")

    # ── 2. Direct medical orders ───────────────────────────────────────────────
    for order in MEDICAL_ORDERS:
        if order in text:
            issues.append(f"Direct medical order detected: '{order}'")

    # ── 3. Missing hedging language ────────────────────────────────────────────
    if not any(hedge in text for hedge in HEDGING_TERMS):
        issues.append("Missing clinical hedging language — response should express uncertainty")

    # ── 4. Filler recommendation detection (Phase 2) ──────────────────────────
    for filler in FILLER_RECOMMENDATIONS:
        if filler in text:
            issues.append(
                f"Filler recommendation detected: '{filler}' — "
                f"replace with specific actionable instruction"
            )

    # ── 5. Actionability check (Phase 2) ──────────────────────────────────────
    # Only check if there IS a recommendations section
    if "actionable next steps" in text or "next steps" in text or "recommendation" in text:
        has_actionable = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in ACTIONABLE_PATTERNS
        )
        if not has_actionable:
            issues.append(
                "No specific actionable recommendation found — "
                "response must specify a concrete clinical action (imaging, test, referral, timeframe)"
            )

    # ── 6. CNN finding omission check (Phase 2) ───────────────────────────────
    if pathology_findings:
        detected_pathologies = _extract_pathology_names_from_findings(pathology_findings)
        for pathology_name in detected_pathologies:
            if pathology_name and pathology_name not in text:
                # Check if at least the discordance note is there
                if "discordance" not in text:
                    issues.append(
                        f"CNN-detected pathology omitted from response: '{pathology_name}' — "
                        f"either acknowledge it or add a Discordance Note"
                    )

    # ── 7. Phantom citation check (NEW) ───────────────────────────────────────
    if evidence:
        response_lower_plain = response.lower()
        for i, e in enumerate(evidence, start=1):
            has_text = bool((e.get("report_text") or "").strip())
            ref_text = f"[r{i}]"
            pattern = rf"\[r{i}\](?!-image)"
            if re.search(pattern, response_lower_plain) and not has_text:
                issues.append(
                    f"Phantom citation [R{i}] — evidence {i} has no text report. "
                    f"Remove this citation or use [R{i}-IMAGE] if image insight exists."
                )

    # ── Result ─────────────────────────────────────────────────────────────────
    if issues:
        logger.warning(f"[SafetyValidator] {len(issues)} issue(s) detected: {issues}")
        return {"passed": False, "issues": issues}

    logger.info("[SafetyValidator] All checks passed")
    return {"passed": True, "issues": []}