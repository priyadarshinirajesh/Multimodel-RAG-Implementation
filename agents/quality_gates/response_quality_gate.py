# agents/quality_gates/response_quality_gate.py

import re
from utils.logger import get_logger

logger = get_logger("ResponseQualityGate")


def _to_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


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

        logger.info(f"[{self.name}] Decision: {decision} (score={score:.2f}, threshold={self.threshold:.2f})")

        return {
            "decision":         decision,
            "score":            score,
            "feedback":         feedback,
            "suggested_action": suggested_action,
        }


class ResponseQualityGate(QualityGate):
    """Quality gate after clinical reasoning.

    Issue 9 FIX: threshold is now a constructor parameter (default 0.7) so
    the graph node can pass in the UI-configured value from MMRAgState:

        gate = ResponseQualityGate(threshold=state.get("response_threshold", 0.7))

    Previously the threshold was hardcoded to 0.7 and the sidebar slider had
    no effect on gate behaviour.
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__(name="ResponseQualityGate", threshold=threshold)

    def evaluate(self, response: str, evidence: list, metrics: dict) -> dict:
        logger.info(f"[{self.name}] Evaluating response quality (threshold={self.threshold:.2f})...")

        groundedness         = _to_float(metrics.get("Groundedness", 0.0))
        completeness         = _to_float(metrics.get("Completeness", 0.0))
        clinical_correctness = _to_float(metrics.get("ClinicalCorrectness", 0.0))

        score = (
            groundedness         * 0.4
            + completeness       * 0.3
            + clinical_correctness * 0.3
        )

        citations   = re.findall(r"\[R\d+\]", response)
        has_structure = (
            "diagnosis" in response.lower()
            and "evidence" in response.lower()
        )

        issues = []
        if groundedness < 0.7:
            issues.append("weak citations")
        if completeness < 0.6:
            issues.append("incomplete sections")
        if clinical_correctness < 0.5:
            issues.append("low clinical accuracy")
        if not has_structure:
            issues.append("missing structure")

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