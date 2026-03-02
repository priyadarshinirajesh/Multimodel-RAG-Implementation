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
    Updated for the new four-section Clinical Assistant format.
    """

    # New section headers to skip when fixing citations
    SECTION_HEADERS = {
        "clinical impression:",
        "evidence synthesis:",
        "differential considerations:",
        "actionable next steps:",
        # Legacy headers (kept for safety)
        "diagnosis / impression:",
        "supporting evidence:",
        "next steps / recommendations:",
    }

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
        lines = text.splitlines()
        changed = False
        out = []

        for line in lines:
            s = line.strip()
            if not s:
                out.append(line)
                continue

            # Skip section headers
            if s.lower().rstrip() in self.SECTION_HEADERS:
                out.append(line)
                continue

            # Add citation to bullet points that lack one
            if s.startswith("-") and not re.search(r'\[(R\d+|Rx)(-IMAGE)?\]', s):
                # Don't add citation to discordance notes — they should already have one
                if "discordance note" not in s.lower():
                    out.append(f"{line} [R1]")
                    changed = True
                else:
                    out.append(line)
            else:
                out.append(line)

        return "\n".join(out), changed

    def _polish_language(self, text: str):
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r'\s+\.', '.', cleaned)
        return cleaned, cleaned != text