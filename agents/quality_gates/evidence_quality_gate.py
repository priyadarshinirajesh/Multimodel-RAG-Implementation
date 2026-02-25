# agents/quality_gates/evidence_quality_gate.py

from utils.logger import get_logger

logger = get_logger("EvidenceQualityGate")


class QualityGate:
    """Base class for all quality gates"""
    
    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold
        self.pass_count = 0
        self.fail_count = 0
    
    def evaluate(self, **kwargs) -> dict:
        """Evaluate quality and return decision."""
        raise NotImplementedError
    
    def _make_decision(self, score: float, feedback: str, suggested_action: str = None) -> dict:
        """Helper to make decision based on score"""
        
        if score >= self.threshold:
            decision = "PASS"
            self.pass_count += 1
        elif score >= self.threshold * 0.8:
            decision = "RETRY"
        else:
            decision = "FAIL"
            self.fail_count += 1
        
        logger.info(f"[{self.name}] Decision: {decision} (score: {score:.2f})")
        
        return {
            "decision": decision,
            "score": score,
            "feedback": feedback,
            "suggested_action": suggested_action
        }


class EvidenceQualityGate(QualityGate):
    """Quality gate after evidence retrieval"""
    
    def __init__(self):
        super().__init__(name="EvidenceQualityGate", threshold=0.4)
    
    def evaluate(self, evidence: list, filter_result: dict, query: str) -> dict:
        logger.info(f"[{self.name}] Evaluating evidence quality...")

        score = float(filter_result.get("quality_score", 0.0))
        filtered_count = len(filter_result.get("filtered_evidence", []))

        # Optional hard fail only when nothing is available
        if filtered_count == 0:
            score = 0.0
            feedback = "No relevant evidence retrieved"
            suggested_action = "Retry retrieval"
            return self._make_decision(score, feedback, suggested_action)

        if score >= self.threshold:
            feedback = f"Evidence relevance is good (avg relevance={score:.2f})"
            suggested_action = None
        else:
            feedback = f"Evidence relevance is low (avg relevance={score:.2f})"
            suggested_action = "Refine query or retrieval settings"

        return self._make_decision(score, feedback, suggested_action)
