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
        super().__init__(name="EvidenceQualityGate", threshold=0.6)
    
    def evaluate(self, evidence: list, filter_result: dict, query: str) -> dict:
        """
        Evaluate evidence quality.
        
        Args:
            evidence: Retrieved evidence
            filter_result: Output from EvidenceQualityVerifier
            query: User query
        """
        logger.info(f"[{self.name}] Evaluating evidence quality...")
        
        quality_score = filter_result.get("quality_score", 0.0)
        filtered_count = len(filter_result.get("filtered_evidence", []))
        needs_adjustment = filter_result.get("needs_retrieval_adjustment", False)
        
        # Calculate gate score
        if filtered_count < 2:
            score = 0.3  # Insufficient evidence
        elif filtered_count > 10:
            score = min(quality_score, 0.7)  # Too much evidence (likely noisy)
        else:
            score = quality_score
        
        # Generate feedback
        if score >= self.threshold:
            feedback = f"Evidence quality is good ({filtered_count} relevant items)"
            suggested_action = None
        elif needs_adjustment:
            feedback = filter_result.get("feedback", "Insufficient evidence")
            suggested_action = "Increase retrieval limit or broaden query"
        else:
            feedback = "Evidence quality is marginal"
            suggested_action = "Proceed but flag for review"
        
        return self._make_decision(score, feedback, suggested_action)