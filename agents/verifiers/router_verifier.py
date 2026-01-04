# agents/verifiers/router_verifier.py

import requests
from utils.logger import get_logger

logger = get_logger("RouterVerifier")

OLLAMA_URL = "http://localhost:11434/api/generate"


class RouterVerifier:
    """Lightweight verifier that checks routing decisions immediately."""
    
    def __init__(self):
        self.confidence_threshold = 0.8
    
    def verify_routing(self, query: str, selected_modalities: list) -> dict:
        """Fast verification of routing decision."""
        
        logger.info(f"[RouterVerifier] Verifying routing for query: {query[:50]}...")
        
        # Rule-based quick checks (fast path)
        quick_check = self._quick_rule_check(query, selected_modalities)
        if quick_check["confidence"] >= self.confidence_threshold:
            logger.info(f"[RouterVerifier] Quick check passed with confidence {quick_check['confidence']:.2f}")
            return quick_check
        
        # LLM-based verification (slow path - only if quick check uncertain)
        logger.info("[RouterVerifier] Running LLM verification...")
        return self._llm_verify_routing(query, selected_modalities)
    
    def _quick_rule_check(self, query: str, selected_modalities: list) -> dict:
        """Rule-based fast verification (no LLM call)"""
        
        query_lower = query.lower()
        
        # Define clear mappings
        XRAY_KEYWORDS = ["chest", "lung", "pulmonary", "pneumonia", "effusion", 
                         "heart", "cardiac", "pleura", "pneumothorax"]
        CT_KEYWORDS = ["pancreas", "pancreatitis", "pancreatic"]
        MRI_KEYWORDS = ["prostate", "pi-rads", "psa"]
        
        # Detect expected modalities
        expected = []
        if any(kw in query_lower for kw in XRAY_KEYWORDS):
            expected.append("XRAY")
        if any(kw in query_lower for kw in CT_KEYWORDS):
            expected.append("CT")
        if any(kw in query_lower for kw in MRI_KEYWORDS):
            expected.append("MRI")
        
        # If no clear keywords, ambiguous query
        if not expected:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "suggested_modalities": selected_modalities,
                "needs_rerun": False,
                "reasoning": "Ambiguous query, LLM verification needed"
            }
        
        # Check if selected matches expected
        selected_set = set(selected_modalities)
        expected_set = set(expected)
        
        if selected_set == expected_set:
            return {
                "is_valid": True,
                "confidence": 0.95,
                "suggested_modalities": selected_modalities,
                "needs_rerun": False,
                "reasoning": "Perfect match with keywords"
            }
        
        # Missing modalities
        missing = expected_set - selected_set
        if missing:
            return {
                "is_valid": False,
                "confidence": 0.6,
                "suggested_modalities": list(expected_set),
                "needs_rerun": True,
                "reasoning": f"Missing modalities: {missing}"
            }
        
        # Extra unnecessary modalities
        extra = selected_set - expected_set
        if extra:
            return {
                "is_valid": True,
                "confidence": 0.75,
                "suggested_modalities": list(expected_set),
                "needs_rerun": False,
                "reasoning": f"Extra modalities: {extra} (not critical)"
            }
        
        return {
            "is_valid": True,
            "confidence": 0.7,
            "suggested_modalities": selected_modalities,
            "needs_rerun": False,
            "reasoning": "Partial match"
        }
    
    def _llm_verify_routing(self, query: str, selected_modalities: list) -> dict:
        """LLM-based verification for complex cases"""
        
        prompt = f"""
You are a medical imaging routing validator.

USER QUERY: "{query}"
SELECTED MODALITIES: {selected_modalities}

AVAILABLE DATASETS:
- XRAY: Chest (lungs, heart, pleura, pneumonia, effusion)
- CT: Pancreas (pancreatitis, pancreatic masses)
- MRI: Prostate (cancer, PI-RADS)

TASK: Validate if routing is correct. Respond ONLY with JSON:

{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "suggested_modalities": ["XRAY", "CT", "MRI"],
    "needs_rerun": true/false,
    "reasoning": "brief explanation"
}}
"""
        
        try:
            payload = {
                "model": "mistral:instruct",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=20)
            response.raise_for_status()
            
            result_text = response.json()["response"].strip()
            
            import json
            result = json.loads(result_text)
            
            logger.info(f"[RouterVerifier] LLM verification: confidence={result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"[RouterVerifier] LLM verification failed: {e}")
            return {
                "is_valid": True,
                "confidence": 0.5,
                "suggested_modalities": selected_modalities,
                "needs_rerun": False,
                "reasoning": "Verification failed, defaulting to selected"
            }