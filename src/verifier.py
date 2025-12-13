# src/verifier.py
import re
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_claims(text):
    claims = [c.strip() for c in re.split(r"[.\n]", text) if c.strip()]
    return claims

def verify_claim_against_evidence(claim, retrieved, threshold=0.40):
    evidence_texts = []
    for r in retrieved:
        evidence_texts.append(r.get("findings", "") or "")
        evidence_texts.append(r.get("impression", "") or "")

    if not any(evidence_texts):
        print("⚠ Empty evidence → verification impossible.")
        return False, None, 0.0

    emb = model.encode([claim] + evidence_texts)
    c = emb[0]
    e = emb[1:]

    sims = np.dot(e, c) / (np.linalg.norm(e, axis=1) * np.linalg.norm(c) + 1e-12)
    best = int(np.argmax(sims))
    score = float(sims[best])
    verified = score >= threshold

    return verified, best // 2, score

def run_verification(text, retrieved):
    print("\n================ VERIFICATION DEBUG ================")
    claims = extract_claims(text)
    print("Extracted Claims:", claims)

    results = []
    for c in claims:
        status = verify_claim_against_evidence(c, retrieved)
        results.append({"claim": c, "verified": status[0], "score": status[2]})
        print(f"CLAIM: {c}\n → Verified: {status[0]}, Score={status[2]:.4f}")

    print("=====================================================\n")
    return results
