# # agents/quality_gates/routing_quality_gate.py

# from utils.logger import get_logger

# logger = get_logger("RoutingQualityGate")


# class QualityGate:
#     """Base class for all quality gates"""
    
#     def __init__(self, name: str, threshold: float = 0.7):
#         self.name = name
#         self.threshold = threshold
#         self.pass_count = 0
#         self.fail_count = 0
    
#     def evaluate(self, **kwargs) -> dict:
#         """Evaluate quality and return decision."""
#         raise NotImplementedError
    
#     def _make_decision(self, score: float, feedback: str, suggested_action: str = None) -> dict:
#         """Helper to make decision based on score"""
        
#         if score >= self.threshold:
#             decision = "PASS"
#             self.pass_count += 1
#         elif score >= self.threshold * 0.8:
#             decision = "RETRY"
#         else:
#             decision = "FAIL"
#             self.fail_count += 1
        
#         logger.info(f"[{self.name}] Decision: {decision} (score: {score:.2f})")
        
#         return {
#             "decision": decision,
#             "score": score,
#             "feedback": feedback,
#             "suggested_action": suggested_action
#         }


# class RoutingQualityGate(QualityGate):
#     """Quality gate after routing decision"""
    
#     def __init__(self):
#         super().__init__(name="RoutingQualityGate", threshold=0.8)
    
#     def evaluate(self, query: str, selected_modalities: list, verification_result: dict) -> dict:
#         """
#         Evaluate routing quality.
        
#         Args:
#             query: User query
#             selected_modalities: Selected modalities
#             verification_result: Output from RouterVerifier
#         """
#         logger.info(f"[{self.name}] Evaluating routing quality...")
        
#         confidence = verification_result.get("confidence", 0.0)
#         is_valid = verification_result.get("is_valid", True)
        
#         # Calculate gate score
#         score = confidence if is_valid else confidence * 0.5
        
#         # Generate feedback
#         if score >= self.threshold:
#             feedback = f"Routing is correct: {selected_modalities}"
#             suggested_action = None
#         elif verification_result.get("needs_rerun"):
#             feedback = f"Routing needs correction: {verification_result.get('reasoning')}"
#             suggested_action = f"Change modalities to: {verification_result.get('suggested_modalities')}"
#         else:
#             feedback = "Routing is acceptable but suboptimal"
#             suggested_action = "Proceed with current modalities"
        
#         return self._make_decision(score, feedback, suggested_action)