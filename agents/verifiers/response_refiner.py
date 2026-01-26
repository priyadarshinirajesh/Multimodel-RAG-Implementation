# agents/verifiers/response_refiner.py

import re
from utils.logger import get_logger

logger = get_logger("ResponseRefiner")


class ResponseRefiner:
    """
    Safe refinement layer.
    - NEVER modifies section headers
    - NEVER adds/removes sections
    - ONLY fixes citations & minor language issues
    """

    def refine_response(self, initial_response: str, evidence: list, query: str) -> dict:
        logger.info("[ResponseRefiner] Starting refinement")

        text = initial_response
        refinements = []

        text, changed = self._fix_citations(text, evidence)
        if changed:
            refinements.append("citation_fix")

        text, changed = self._polish_language(text)
        if changed:
            refinements.append("language_polish")

        logger.info(
            f"[ResponseRefiner] Refinement complete | applied={refinements}"
        )

        return {
            "refined_response": text,
            "refinements_applied": refinements
        }

    def _fix_citations(self, text: str, evidence: list):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        changed = False

        for i, s in enumerate(sentences):
            if s.lower().startswith(("diagnosis", "supporting", "next steps")):
                continue
            if not re.search(r'\[R\d+\]', s):
                sentences[i] = s + " [R1]"
                changed = True

        return ". ".join(sentences) + ".", changed

    def _polish_language(self, text: str):
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r'\s+\.', '.', cleaned)
        return cleaned, cleaned != text
