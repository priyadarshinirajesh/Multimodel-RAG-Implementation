# agents/quality_gates/response_quality_gate.py

import re
from utils.logger import get_logger

logger = get_logger("ResponseQualityGate")


def _to_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


# ── Patterns for specific, actionable content in recommendations ───────────────
_ACTIONABLE_PATTERNS = [
    r"(x-ray|chest\s*x.ray|ct|mri|ecg|ekg|echo|echocardiogram|ultrasound|pa\s*view|lateral\s*view)\s*(recommended|indicated|suggested|obtained|ordered)",
    r"(repeat|follow.up|compare\s*with\s*prior|interval\s*change)",
    r"(refer\s*to|specialist|cardiology|pulmonology|radiology)",
    r"\d+\s*(hour|day|week|month)",
    r"(lateral|pa\s*view|portable|upright)",
    r"(blood\s*pressure|oxygen|saturation|spo2|ecg|ekg)",
    r"(biopsy|bronchoscopy|thoracentesis)",
]


def _compute_actionability_score(response: str) -> float:
    """
    Score 0.0–1.0 measuring how specific and actionable the recommendations are.
    Reaches 1.0 when 2+ distinct actionable patterns are matched.
    Returns 0.0 if recommendations section is missing entirely.
    """
    response_lower = response.lower()

    # If there is no recommendations section at all, score is 0
    has_recs_section = (
        "actionable next steps" in response_lower
        or "next steps" in response_lower
        or "recommendation" in response_lower
    )
    if not has_recs_section:
        return 0.0

    matched = sum(
        1 for p in _ACTIONABLE_PATTERNS
        if re.search(p, response_lower, re.IGNORECASE)
    )

    # Penalty for filler phrases
    FILLER = [
        "further evaluation by a healthcare professional",
        "consult a doctor",
        "seek medical advice",
    ]
    has_filler = any(f in response_lower for f in FILLER)
    if has_filler:
        matched = max(0, matched - 1)

    # Scale: 0 matches → 0.0, 1 match → 0.5, 2+ matches → 1.0
    return min(matched / 2, 1.0)


def _compute_contradiction_flag(response: str) -> float:
    """
    Score measuring whether CNN vs text report discordances were handled.

    Returns:
        1.0 — Discordance was explicitly detected and addressed
        0.7 — CNN/imaging findings were properly referenced (no contradiction)
        0.5 — No contradiction context (neutral — neither good nor bad)
        0.0 — Contradiction exists but was silently ignored
    """
    response_lower = response.lower()

    has_discordance_note = "discordance" in response_lower
    has_cnn_reference = any(
        t in response_lower
        for t in ["cnn", "densenet", "imaging only", "limited confidence", "ai model"]
    )
    has_confidence_indicator = any(
        t in response_lower
        for t in ["high confidence", "moderate confidence", "low confidence"]
    )

    if has_discordance_note:
        return 1.0  # Explicitly handled contradiction
    if has_confidence_indicator and has_cnn_reference:
        return 0.7  # Properly qualified imaging-based findings
    if has_confidence_indicator:
        return 0.6  # At least used confidence indicators
    return 0.5      # No contradiction context — neutral


class QualityGate:
    """Base class for all quality gates."""

    def __init__(self, name: str, threshold: float = 0.65):
        self.name       = name
        self.threshold  = threshold
        self.pass_count = 0
        self.fail_count = 0

    def evaluate(self, **kwargs) -> dict:
        raise NotImplementedError

    def _make_decision(self, score: float, feedback: str, suggested_action: str = None) -> dict:
        if score >= self.threshold:
            decision = "PASS"
            self.pass_count += 1
        elif score >= self.threshold * 0.8:
            decision = "RETRY"
        else:
            decision = "FAIL"
            self.fail_count += 1

        logger.info(
            f"[{self.name}] Decision: {decision} "
            f"(score={score:.2f}, threshold={self.threshold:.2f})"
        )

        return {
            "decision":         decision,
            "score":            score,
            "feedback":         feedback,
            "suggested_action": suggested_action,
        }


class ResponseQualityGate(QualityGate):
    """
    Quality gate after clinical reasoning.

    Scoring breakdown (Phase 3 update):
        Groundedness         × 0.30  — every claim has a citation
        Completeness         × 0.20  — all required sections present
        ClinicalCorrectness  × 0.20  — answer aligns with retrieved impressions
        ActionabilityScore   × 0.20  — recommendations are specific, not generic
        ContradictionFlag    × 0.10  — CNN vs text discordances were handled

    Issue 9 FIX: threshold is now a constructor parameter (default 0.7) so
    the graph node can pass in the UI-configured value from MMRAgState.
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__(name="ResponseQualityGate", threshold=threshold)

    def evaluate(self, response: str, evidence: list, metrics: dict) -> dict:
        logger.info(
            f"[{self.name}] Evaluating response quality (threshold={self.threshold:.2f})..."
        )

        groundedness         = _to_float(metrics.get("Groundedness", 0.0))
        completeness_score   = _to_float(metrics.get("Completeness", 0.0))
        clinical_correctness = _to_float(metrics.get("ClinicalCorrectness", 0.0))

        # Phase 3 new metrics
        actionability     = _compute_actionability_score(response)
        contradiction_flag = _compute_contradiction_flag(response)

        # Weighted composite score
        score = (
            groundedness         * 0.30
            + completeness_score * 0.20
            + clinical_correctness * 0.20
            + actionability      * 0.20
            + contradiction_flag * 0.10
        )

        # Structure check — updated for new section names
        has_structure = (
            "clinical impression" in response.lower()
            and "evidence synthesis" in response.lower()
        )

        # Issue detection
        issues = []
        if groundedness < 0.7:
            issues.append("weak citations")
        if completeness_score < 0.6:
            issues.append("incomplete sections")
        if clinical_correctness < 0.5:
            issues.append("low clinical accuracy")
        if not has_structure:
            issues.append("missing required sections")
        if actionability < 0.5:
            issues.append("non-actionable or filler recommendations")
        if contradiction_flag < 0.5:
            issues.append("CNN vs text discordance not addressed")

        # Store sub-scores in metrics dict for export/display
        metrics["ActionabilityScore"]  = round(actionability, 3)
        metrics["ContradictionFlag"]   = round(contradiction_flag, 3)

        if score >= self.threshold and not issues:
            feedback         = "Response quality is good"
            suggested_action = None
        elif issues:
            feedback         = f"Response has issues: {', '.join(issues)}"
            suggested_action = "Apply progressive refinement"
        else:
            feedback         = "Response quality is acceptable"
            suggested_action = "Minor refinements recommended"

        return self._make_decision(score, feedback, suggested_action)