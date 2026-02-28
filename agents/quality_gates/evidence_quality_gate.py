# agents/quality_gates/evidence_quality_gate.py

from utils.logger import get_logger

logger = get_logger("EvidenceQualityGate")


class QualityGate:
    """Base class for all quality gates."""

    def __init__(self, name: str, threshold: float = 0.7):
        self.name      = name
        self.threshold = threshold
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


class EvidenceQualityGate(QualityGate):
    """Quality gate after evidence retrieval.

    Issue 9 FIX: threshold is now a constructor parameter (default 0.4) so
    the graph node can pass in the UI-configured value from MMRAgState:

        gate = EvidenceQualityGate(threshold=state.get("evidence_threshold", 0.4))

    Previously the threshold was hardcoded to 0.4 and the sidebar slider had
    no effect on gate behaviour.
    """

    def __init__(self, threshold: float = 0.4):
        super().__init__(name="EvidenceQualityGate", threshold=threshold)

    def evaluate(self, evidence: list, filter_result: dict, query: str) -> dict:
        logger.info(f"[{self.name}] Evaluating evidence quality (threshold={self.threshold:.2f})...")

        score          = float(filter_result.get("quality_score", 0.0))
        filtered_count = len(filter_result.get("filtered_evidence", []))

        if filtered_count == 0:
            return self._make_decision(
                0.0,
                "No relevant evidence retrieved",
                "Retry retrieval",
            )

        if score >= self.threshold:
            feedback         = f"Evidence relevance is good (avg relevance={score:.2f})"
            suggested_action = None
        else:
            feedback         = f"Evidence relevance is low (avg relevance={score:.2f})"
            suggested_action = "Refine query or retrieval settings"

        return self._make_decision(score, feedback, suggested_action)