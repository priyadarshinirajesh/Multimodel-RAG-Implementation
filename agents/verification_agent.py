# # agents/verification_agent.py

# import requests
# from utils.logger import get_logger

# logger = get_logger("VerificationAgent")

# OLLAMA_URL = "http://localhost:11434/api/generate"

# class VerificationAgent:
#     """
#     Verification Agent validates the clinical reasoning pipeline
#     and suggests improvements or re-routing if necessary.
#     """
    
#     def __init__(self):
#         self.verification_history = []
    
#     def verify_pipeline(
#         self,
#         query: str,
#         selected_modalities: list,
#         evidence: list,
#         final_answer: str,
#         metrics: dict
#     ):
#         """
#         Main verification function that checks:
#         1. Modality routing correctness
#         2. Evidence relevance and completeness
#         3. Clinical response quality
#         4. Citation and groundedness
        
#         Returns:
#         - verification_result: dict with pass/fail status
#         - improvement_suggestions: list of actionable improvements
#         - requires_rerun: bool indicating if pipeline needs re-execution
#         """
        
#         logger.info("=" * 60)
#         logger.info("VERIFICATION AGENT: Starting pipeline verification")
#         logger.info("=" * 60)
        
#         verification_result = {
#             "modality_routing": self._verify_modality_routing(query, selected_modalities),
#             "evidence_quality": self._verify_evidence_quality(query, evidence),
#             "clinical_response": self._verify_clinical_response(final_answer, evidence, metrics),
#             "citation_check": self._verify_citations(final_answer, evidence),
#             "overall_pass": False,
#             "confidence_score": 0.0
#         }
        
#         # Calculate overall confidence
#         confidence_components = [
#             verification_result["modality_routing"]["score"],
#             verification_result["evidence_quality"]["score"],
#             verification_result["clinical_response"]["score"],
#             verification_result["citation_check"]["score"]
#         ]
#         verification_result["confidence_score"] = sum(confidence_components) / len(confidence_components)
#         verification_result["overall_pass"] = verification_result["confidence_score"] >= 0.7
        
#         # Generate improvement suggestions
#         improvement_suggestions = self._generate_improvements(verification_result, query, evidence)
        
#         # Determine if re-run is needed
#         requires_rerun = self._should_rerun(verification_result, improvement_suggestions)
        
#         # Log verification results
#         self._log_verification(verification_result, improvement_suggestions, requires_rerun)
        
#         return {
#             "verification_result": verification_result,
#             "improvement_suggestions": improvement_suggestions,
#             "requires_rerun": requires_rerun
#         }
    
#     def _verify_modality_routing(self, query: str, selected_modalities: list):
#         """Verify if the correct modalities were selected for the query"""
        
#         logger.info("[Verification] Checking modality routing...")
        
#         prompt = f"""
# You are a medical imaging routing validator.

# USER QUERY: "{query}"
# SELECTED MODALITIES: {selected_modalities}

# AVAILABLE DATASETS:
# - XRAY: Chest X-rays (lungs, heart, pleura, pneumonia, effusion)
# - CT: Pancreas CT (pancreatitis, pancreatic masses)
# - MRI: Prostate MRI (prostate cancer, PI-RADS lesions)

# TASK:
# Evaluate if the selected modalities are appropriate for this query.

# OUTPUT (JSON format only):
# {{
#     "is_correct": true/false,
#     "score": 0.0-1.0,
#     "reasoning": "brief explanation",
#     "missing_modalities": ["modality1", ...] or [],
#     "unnecessary_modalities": ["modality1", ...] or []
# }}
# """
        
#         try:
#             payload = {
#                 "model": "deepseek-r1:7b",
#                 "prompt": prompt,
#                 "stream": False
#             }
            
#             response = requests.post(OLLAMA_URL, json=payload, timeout=30)
#             response.raise_for_status()
            
#             result_text = response.json()["response"].strip()
            
#             # Parse JSON response
#             import json
#             result = json.loads(result_text)
            
#             logger.info(f"[Verification] Modality routing score: {result['score']}")
            
#             return result
            
#         except Exception as e:
#             logger.error(f"[Verification] Modality routing check failed: {e}")
#             return {
#                 "is_correct": True,
#                 "score": 0.5,
#                 "reasoning": "Verification failed, assuming correctness",
#                 "missing_modalities": [],
#                 "unnecessary_modalities": []
#             }
    
#     def _verify_evidence_quality(self, query: str, evidence: list):
#         """Verify if retrieved evidence is relevant and sufficient"""
        
#         logger.info("[Verification] Checking evidence quality...")
        
#         if not evidence:
#             return {
#                 "is_sufficient": False,
#                 "score": 0.0,
#                 "reasoning": "No evidence retrieved",
#                 "relevant_count": 0,
#                 "total_count": 0
#             }
        
#         # Count evidence with images vs text-only
#         evidence_with_images = sum(1 for e in evidence if e.get("has_image"))
#         text_only = len(evidence) - evidence_with_images
        
#         # Check for distractor evidence
#         distractor_keywords = ["normal", "no acute findings", "unremarkable"]
#         distractors = sum(
#             1 for e in evidence
#             if any(kw in e.get("report_text", "").lower() for kw in distractor_keywords)
#         )
        
#         prompt = f"""
# You are a clinical evidence quality assessor.

# CLINICAL QUERY: "{query}"

# RETRIEVED EVIDENCE SUMMARY:
# - Total evidence items: {len(evidence)}
# - Evidence with images: {evidence_with_images}
# - Text-only reports: {text_only}
# - Potential distractors: {distractors}

# SAMPLE EVIDENCE:
# {self._format_evidence_sample(evidence[:3])}

# TASK:
# Assess if this evidence is sufficient and relevant to answer the query.

# OUTPUT (JSON format only):
# {{
#     "is_sufficient": true/false,
#     "score": 0.0-1.0,
#     "reasoning": "brief explanation",
#     "relevant_count": <number>,
#     "irrelevant_count": <number>,
#     "needs_more_evidence": true/false
# }}
# """
        
#         try:
#             payload = {
#                 "model": "deepseek-r1:7b",
#                 "prompt": prompt,
#                 "stream": False
#             }
            
#             response = requests.post(OLLAMA_URL, json=payload, timeout=30)
#             response.raise_for_status()
            
#             result_text = response.json()["response"].strip()
            
#             import json
#             result = json.loads(result_text)
            
#             logger.info(f"[Verification] Evidence quality score: {result['score']}")
            
#             return result
            
#         except Exception as e:
#             logger.error(f"[Verification] Evidence quality check failed: {e}")
#             return {
#                 "is_sufficient": True,
#                 "score": 0.6,
#                 "reasoning": "Verification failed, assuming adequacy",
#                 "relevant_count": len(evidence),
#                 "irrelevant_count": 0,
#                 "needs_more_evidence": False
#             }
    
#     def _verify_clinical_response(self, final_answer: str, evidence: list, metrics: dict):
#         """Verify clinical response quality and structure"""
        
#         logger.info("[Verification] Checking clinical response...")
        
#         # Check structure
#         required_sections = ["diagnosis", "supporting evidence", "next steps"]
#         has_structure = all(
#             section in final_answer.lower()
#             for section in required_sections
#         )
        
#         # Check metrics
#         groundedness = metrics.get("Groundedness", 0)
#         completeness = metrics.get("Completeness", 0)
#         clinical_correctness = metrics.get("ClinicalCorrectness", 0)
        
#         prompt = f"""
# You are a clinical response quality validator.

# CLINICAL RESPONSE:
# {final_answer}

# QUALITY METRICS:
# - Groundedness: {groundedness}
# - Completeness: {completeness}
# - Clinical Correctness: {clinical_correctness}
# - Has proper structure: {has_structure}

# TASK:
# Evaluate the quality of this clinical response.

# OUTPUT (JSON format only):
# {{
#     "is_acceptable": true/false,
#     "score": 0.0-1.0,
#     "reasoning": "brief explanation",
#     "missing_elements": ["element1", ...] or [],
#     "quality_issues": ["issue1", ...] or []
# }}
# """
        
#         try:
#             payload = {
#                 "model": "deepseek-r1:7b",
#                 "prompt": prompt,
#                 "stream": False
#             }
            
#             response = requests.post(OLLAMA_URL, json=payload, timeout=30)
#             response.raise_for_status()
            
#             result_text = response.json()["response"].strip()
            
#             import json
#             result = json.loads(result_text)
            
#             logger.info(f"[Verification] Clinical response score: {result['score']}")
            
#             return result
            
#         except Exception as e:
#             logger.error(f"[Verification] Clinical response check failed: {e}")
#             return {
#                 "is_acceptable": True,
#                 "score": 0.7,
#                 "reasoning": "Verification failed, assuming acceptability",
#                 "missing_elements": [],
#                 "quality_issues": []
#             }
    
#     def _verify_citations(self, final_answer: str, evidence: list):
#         """Verify citation correctness and completeness"""
        
#         logger.info("[Verification] Checking citations...")
        
#         import re
        
#         # Extract all citations
#         citations = re.findall(r'\[R(\d+)(-IMAGE)?\]', final_answer)
#         cited_indices = set(int(c[0]) for c in citations)
        
#         # Check for hallucinated citations
#         max_evidence_idx = len(evidence)
#         hallucinated = [idx for idx in cited_indices if idx > max_evidence_idx]
        
#         # Check for missing citations (factual statements without citations)
#         sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
#         factual_sentences = [
#             s for s in sentences
#             if not any(skip in s.lower() for skip in ['diagnosis', 'supporting evidence', 'next steps'])
#         ]
#         uncited_sentences = [
#             s for s in factual_sentences
#             if not re.search(r'\[R\d+(-IMAGE)?\]', s)
#         ]
        
#         citation_coverage = 1 - (len(uncited_sentences) / len(factual_sentences)) if factual_sentences else 1.0
        
#         score = citation_coverage
#         if hallucinated:
#             score *= 0.5  # Heavy penalty for hallucinated citations
        
#         return {
#             "is_valid": len(hallucinated) == 0 and citation_coverage >= 0.8,
#             "score": round(score, 3),
#             "reasoning": f"Citation coverage: {citation_coverage:.2%}, Hallucinated: {len(hallucinated)}",
#             "hallucinated_citations": hallucinated,
#             "uncited_count": len(uncited_sentences)
#         }
    
#     def _generate_improvements(self, verification_result: dict, query: str, evidence: list):
#         """Generate actionable improvement suggestions"""
        
#         suggestions = []
        
#         # Modality routing improvements
#         if verification_result["modality_routing"]["score"] < 0.7:
#             missing = verification_result["modality_routing"].get("missing_modalities", [])
#             unnecessary = verification_result["modality_routing"].get("unnecessary_modalities", [])
            
#             if missing:
#                 suggestions.append({
#                     "agent": "modality_router",
#                     "action": "add_modalities",
#                     "details": f"Add modalities: {missing}",
#                     "priority": "HIGH"
#                 })
            
#             if unnecessary:
#                 suggestions.append({
#                     "agent": "modality_router",
#                     "action": "remove_modalities",
#                     "details": f"Remove unnecessary modalities: {unnecessary}",
#                     "priority": "MEDIUM"
#                 })
        
#         # Evidence quality improvements
#         if verification_result["evidence_quality"]["score"] < 0.6:
#             if verification_result["evidence_quality"].get("needs_more_evidence"):
#                 suggestions.append({
#                     "agent": "retrieval_agents",
#                     "action": "increase_retrieval_limit",
#                     "details": "Increase retrieval limit from 5 to 7",
#                     "priority": "HIGH"
#                 })
        
#         # Clinical response improvements
#         if verification_result["clinical_response"]["score"] < 0.7:
#             missing_elements = verification_result["clinical_response"].get("missing_elements", [])
            
#             if missing_elements:
#                 suggestions.append({
#                     "agent": "clinical_reasoning",
#                     "action": "regenerate_with_structure",
#                     "details": f"Missing sections: {missing_elements}",
#                     "priority": "HIGH"
#                 })
        
#         # Citation improvements
#         if verification_result["citation_check"]["score"] < 0.8:
#             if verification_result["citation_check"].get("hallucinated_citations"):
#                 suggestions.append({
#                     "agent": "clinical_reasoning",
#                     "action": "fix_citations",
#                     "details": "Remove hallucinated citations and re-cite properly",
#                     "priority": "CRITICAL"
#                 })
            
#             if verification_result["citation_check"].get("uncited_count", 0) > 0:
#                 suggestions.append({
#                     "agent": "clinical_reasoning",
#                     "action": "add_missing_citations",
#                     "details": f"Add citations to {verification_result['citation_check']['uncited_count']} uncited statements",
#                     "priority": "HIGH"
#                 })
        
#         return suggestions
    
#     def _should_rerun(self, verification_result: dict, suggestions: list):
#         """Determine if pipeline should be re-executed"""
        
#         # Critical failures require immediate rerun
#         critical_suggestions = [s for s in suggestions if s["priority"] == "CRITICAL"]
#         if critical_suggestions:
#             return True
        
#         # Low confidence score requires rerun
#         if verification_result["confidence_score"] < 0.6:
#             return True
        
#         # Multiple high-priority issues require rerun
#         high_priority_count = len([s for s in suggestions if s["priority"] == "HIGH"])
#         if high_priority_count >= 2:
#             return True
        
#         return False
    
#     def _format_evidence_sample(self, evidence_sample: list):
#         """Format evidence for LLM prompt"""
#         formatted = []
#         for idx, e in enumerate(evidence_sample, 1):
#             formatted.append(
#                 f"Evidence {idx}: [{e['modality']}] {e['report_text'][:150]}..."
#             )
#         return "\n".join(formatted)
    
#     def _log_verification(self, result: dict, suggestions: list, requires_rerun: bool):
#         """Log verification results"""
        
#         logger.info("=" * 60)
#         logger.info("VERIFICATION RESULTS")
#         logger.info("=" * 60)
#         logger.info(f"Overall Pass: {result['overall_pass']}")
#         logger.info(f"Confidence Score: {result['confidence_score']:.2%}")
#         logger.info(f"Requires Re-run: {requires_rerun}")
#         logger.info("")
#         logger.info("Component Scores:")
#         logger.info(f"  - Modality Routing: {result['modality_routing']['score']:.2f}")
#         logger.info(f"  - Evidence Quality: {result['evidence_quality']['score']:.2f}")
#         logger.info(f"  - Clinical Response: {result['clinical_response']['score']:.2f}")
#         logger.info(f"  - Citation Check: {result['citation_check']['score']:.2f}")
        
#         if suggestions:
#             logger.info("")
#             logger.info(f"Improvement Suggestions ({len(suggestions)}):")
#             for i, s in enumerate(suggestions, 1):
#                 logger.info(f"  {i}. [{s['priority']}] {s['agent']}: {s['details']}")
        
#         logger.info("=" * 60)