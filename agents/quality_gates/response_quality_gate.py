# agents/quality_gates/response_quality_gate.py

import re
from utils.logger import get_logger

logger = get_logger("ResponseQualityGate")


class QualityGate:
    """Base class for all quality gates"""
    
    def __init__(self, name: str, threshold: float = 0.65):
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


class ResponseQualityGate(QualityGate):
    """Quality gate after clinical reasoning"""
    
    def __init__(self):
        super().__init__(name="ResponseQualityGate", threshold=0.7)
    
    def evaluate(self, response: str, evidence: list, metrics: dict) -> dict:
        """
        Evaluate response quality.
        
        Args:
            response: Generated clinical response
            evidence: Evidence used
            metrics: Evaluation metrics
        """
        logger.info(f"[{self.name}] Evaluating response quality...")
        
        # Extract key metrics
        groundedness = metrics.get("Groundedness", 0.0)
        completeness = metrics.get("Completeness", 0.0)
        clinical_correctness = metrics.get("ClinicalCorrectness", 0.0)
        
        # Calculate composite score
        score = (
            groundedness * 0.4 +
            completeness * 0.3 +
            clinical_correctness * 0.3
        )
        
        # Check for critical issues
        citations = re.findall(r'\[R\d+\]', response)
        has_structure = "diagnosis" in response.lower() and "evidence" in response.lower()
        
        # Generate feedback
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
            feedback = "Response quality is good"
            suggested_action = None
        elif issues:
            feedback = f"Response has issues: {', '.join(issues)}"
            suggested_action = "Apply progressive refinement"
        else:
            feedback = "Response quality is acceptable"
            suggested_action = "Minor refinements recommended"
        
        return self._make_decision(score, feedback, suggested_action)