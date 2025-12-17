# src/agents/verifier_agent.py

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


class VerifierAgent:
    def verify(self, answer, evidence):
        claims = [c.strip() for c in answer.split(".") if len(c.strip()) > 5]

        correction_needed = False
        correction = ""

        for claim in claims:
            best = 0

            for e in evidence:
                txt = f"{e['findings']} {e['impression']}"
                score = util.cos_sim(model.encode(claim), model.encode(txt))
                best = max(best, float(score))

            if best < 0.20:
                correction_needed = True
                correction += f"- Unsupported: '{claim}'\n"

        return correction_needed, correction
