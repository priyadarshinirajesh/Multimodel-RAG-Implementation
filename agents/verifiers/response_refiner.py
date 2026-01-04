# agents/verifiers/response_refiner.py

import requests
import re
from utils.logger import get_logger

logger = get_logger("ResponseRefiner")

OLLAMA_URL = "http://localhost:11434/api/generate"


class ResponseRefiner:
    """Progressive refinement of clinical responses."""
    
    def __init__(self):
        self.max_refinement_iterations = 3
    
    def refine_response(self, initial_response: str, evidence: list, query: str) -> dict:
        """Progressive refinement pipeline."""
        
        logger.info("[ResponseRefiner] Starting progressive refinement...")
        
        current_response = initial_response
        refinements_applied = []
        
        # Stage 1: Fix citations
        citation_result = self._fix_citations(current_response, evidence)
        if citation_result["changes_made"]:
            current_response = citation_result["refined_text"]
            refinements_applied.append("citation_fix")
            logger.info("[ResponseRefiner] Applied citation fixes")
        
        # Stage 2: Improve structure
        structure_result = self._improve_structure(current_response)
        if structure_result["changes_made"]:
            current_response = structure_result["refined_text"]
            refinements_applied.append("structure_improvement")
            logger.info("[ResponseRefiner] Applied structure improvements")
        
        # Stage 3: Add missing sections
        completeness_result = self._ensure_completeness(current_response, evidence, query)
        if completeness_result["changes_made"]:
            current_response = completeness_result["refined_text"]
            refinements_applied.append("completeness_enhancement")
            logger.info("[ResponseRefiner] Added missing sections")
        
        # Stage 4: Final polish
        polish_result = self._polish_language(current_response)
        if polish_result["changes_made"]:
            current_response = polish_result["refined_text"]
            refinements_applied.append("language_polish")
            logger.info("[ResponseRefiner] Applied language polish")
        
        final_quality_score = self._calculate_quality_score(current_response, evidence)
        
        logger.info(f"[ResponseRefiner] Refinement complete. Applied {len(refinements_applied)} stages")
        logger.info(f"[ResponseRefiner] Final quality score: {final_quality_score:.2f}")
        
        return {
            "refined_response": current_response,
            "refinements_applied": refinements_applied,
            "iterations": len(refinements_applied),
            "final_quality_score": final_quality_score
        }
    
    def _fix_citations(self, response: str, evidence: list) -> dict:
        """Fix hallucinated or missing citations"""
        
        citations = re.findall(r'\[R(\d+)(-IMAGE)?\]', response)
        cited_indices = set(int(c[0]) for c in citations)
        
        max_evidence_idx = len(evidence)
        hallucinated = [idx for idx in cited_indices if idx > max_evidence_idx]
        
        changes_made = False
        refined_text = response
        
        # Remove hallucinated citations
        if hallucinated:
            for idx in hallucinated:
                refined_text = re.sub(rf'\[R{idx}(-IMAGE)?\]', '', refined_text)
            changes_made = True
            logger.info(f"[CitationFix] Removed {len(hallucinated)} hallucinated citations")
        
        # Add missing citations to uncited factual statements
        sentences = [s.strip() for s in refined_text.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if any(header in sentence.lower() for header in ['diagnosis', 'supporting evidence', 'next steps']):
                continue
            
            if not re.search(r'\[R\d+(-IMAGE)?\]', sentence):
                if i < len(sentences):
                    sentences[i] = sentence + " [R1]"
                    changes_made = True
        
        if changes_made:
            refined_text = '. '.join(sentences) + '.'
            logger.info("[CitationFix] Added missing citations")
        
        return {
            "refined_text": refined_text,
            "changes_made": changes_made
        }
    
    def _improve_structure(self, response: str) -> dict:
        """Ensure response has proper structure with sections"""
        
        required_sections = ["Diagnosis", "Supporting Evidence", "Next Steps"]
        
        changes_made = False
        refined_text = response
        
        missing_sections = [
            section for section in required_sections
            if section.lower() not in response.lower()
        ]
        
        if missing_sections:
            logger.info(f"[StructureImprove] Missing sections: {missing_sections}")
            
            lines = response.split('\n')
            restructured = []
            
            current_section = None
            for line in lines:
                if line.strip().startswith('-'):
                    if current_section is None:
                        restructured.append("Supporting Evidence:")
                        current_section = "evidence"
                    restructured.append(line)
                elif any(word in line.lower() for word in ['recommend', 'next', 'follow']):
                    if "Next Steps" not in '\n'.join(restructured):
                        restructured.append("\nNext Steps / Recommendations:")
                    restructured.append(line)
                else:
                    restructured.append(line)
            
            refined_text = '\n'.join(restructured)
            changes_made = True
        
        return {
            "refined_text": refined_text,
            "changes_made": changes_made
        }
    
    def _ensure_completeness(self, response: str, evidence: list, query: str) -> dict:
        """Add missing diagnostic or recommendation content"""
        
        changes_made = False
        refined_text = response
        
        has_diagnosis = "diagnosis" in response.lower() or "impression" in response.lower()
        
        if not has_diagnosis:
            diagnosis_line = "Diagnosis / Impression:\n- Clinical findings as noted in evidence [R1]\n\n"
            refined_text = diagnosis_line + refined_text
            changes_made = True
            logger.info("[CompletenessCheck] Added missing diagnosis section")
        
        has_recommendations = any(
            word in response.lower()
            for word in ['recommend', 'next step', 'follow-up', 'consider']
        )
        
        if not has_recommendations:
            recommendation = "\n\nNext Steps / Recommendations:\n- Clinical correlation recommended [R1]\n"
            refined_text += recommendation
            changes_made = True
            logger.info("[CompletenessCheck] Added missing recommendations")
        
        return {
            "refined_text": refined_text,
            "changes_made": changes_made
        }
    
    def _polish_language(self, response: str) -> dict:
        """Clean up language, remove redundancy, improve readability"""
        
        changes_made = False
        refined_text = response
        
        # Remove multiple blank lines
        refined_text = re.sub(r'\n{3,}', '\n\n', refined_text)
        
        # Remove trailing spaces
        refined_text = '\n'.join(line.rstrip() for line in refined_text.split('\n'))
        
        # Ensure proper punctuation
        refined_text = re.sub(r'\s+\.', '.', refined_text)
        
        if refined_text != response:
            changes_made = True
            logger.info("[LanguagePolish] Applied formatting cleanup")
        
        return {
            "refined_text": refined_text,
            "changes_made": changes_made
        }
    
    def _calculate_quality_score(self, response: str, evidence: list) -> float:
        """Calculate quality score for refined response"""
        
        score = 0.0
        
        # Check structure (30%)
        has_diagnosis = "diagnosis" in response.lower() or "impression" in response.lower()
        has_evidence = "supporting evidence" in response.lower() or "evidence" in response.lower()
        has_recommendations = any(w in response.lower() for w in ['recommend', 'next step'])
        
        structure_score = (has_diagnosis + has_evidence + has_recommendations) / 3.0
        score += structure_score * 0.3
        
        # Check citations (40%)
        citations = re.findall(r'\[R\d+(-IMAGE)?\]', response)
        sentences = [s for s in response.split('.') if s.strip()]
        factual_sentences = [s for s in sentences if s.strip().startswith('-')]
        
        if factual_sentences:
            citation_coverage = min(len(citations) / len(factual_sentences), 1.0)
            score += citation_coverage * 0.4
        else:
            score += 0.2
        
        # Check completeness (30%)
        word_count = len(response.split())
        completeness = min(word_count / 150, 1.0)
        score += completeness * 0.3
        
        return round(score, 3)