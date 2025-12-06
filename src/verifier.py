# src/verifier.py
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small sentence-transformer (CPU)
SENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
smodel = SentenceTransformer(SENT_MODEL)

def extract_claims(text):
    """
    Very simple claim extraction: split into sentences and ignore 'Final Answer' label lines.
    """
    lines = [l.strip() for l in re.split(r"[.\n]+", text) if l.strip()]
    # filter out obvious boilerplate
    return [l for l in lines if l.lower() not in ("final answer","task:")]

def verify_claim_against_evidence(claim, retrieved_evidence, threshold=0.45):
    """
    Use semantic similarity: embed claim and evidence sentences, compute max similarity.
    Returns (verified_bool, best_supporting_evidence_idx, score)
    """
    evid_texts = []
    for item in retrieved_evidence:
        evid_texts.append(item.get("findings","") or "")
        evid_texts.append(item.get("impression","") or "")
    if len(evid_texts) == 0:
        return False, None, 0.0

    embeddings = smodel.encode([claim] + evid_texts, convert_to_numpy=True)
    c_emb = embeddings[0]
    e_embs = embeddings[1:]
    # compute cosine sims
    e_norms = np.linalg.norm(e_embs, axis=1) + 1e-12
    c_norm = np.linalg.norm(c_emb) + 1e-12
    sims = (e_embs @ c_emb) / (e_norms * c_norm)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    evidence_item_idx = best_idx // 2  # each item added 2 entries (findings, impression)
    verified = best_score >= threshold
    return verified, evidence_item_idx, best_score

def run_verification(reasoner_text, retrieved):
    claims = extract_claims(reasoner_text)
    results = []
    for c in claims:
        verified, idx, score = verify_claim_against_evidence(c, retrieved)
        results.append({"claim": c, "verified": verified, "evidence_idx": idx, "score": score})
    return results

if __name__ == "__main__":
    retrieved = [
        {"filename":"a.png", "findings":"Cardiac silhouette enlarged.", "impression":"Cardiomegaly."},
        {"filename":"b.png", "findings":"Lungs clear.", "impression":"No acute disease."}
    ]
    reasoner_out = "Step 1: The cardiac silhouette is enlarged [R1]. Final Answer: Likely cardiomegaly [R1]."
    print(run_verification(reasoner_out, retrieved))
