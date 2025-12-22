# src/evaluation.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --------------------------
# 1. Semantic similarity
# --------------------------
def semantic_similarity(pred, gt):
    if not gt.strip():
        return 0.0
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gt = model.encode(gt, convert_to_tensor=True)
    return float(util.cos_sim(emb_pred, emb_gt))

# --------------------------
# 2. Faithfulness Score
# --------------------------
def faithfulness_score(pred, retrieved):
    """
    Faithfulness: checks whether the prediction is grounded
    in retrieved clinical evidence (findings + impressions).
    """
    if not retrieved:
        return 0.0

    evidences = []
    for r in retrieved:
        f = r.get("findings", "")
        imp = r.get("impression", "")
        if f: evidences.append(f)
        if imp: evidences.append(imp)

    if not evidences:
        return 0.0

    emb_pred = model.encode(pred, convert_to_tensor=True)
    evidence_emb = model.encode(evidences, convert_to_tensor=True)

    sims = util.cos_sim(emb_pred, evidence_emb)[0]  
    return float(sims.mean())

# --------------------------
# 3. Hallucination Rate
# --------------------------

def hallucination_rate(prediction, retrieved, threshold=0.20):
    """
    Hallucination Rate:
    Fraction of claims in prediction that are NOT supported
    by retrieved findings + impressions.
    """

    # Split answer into claims (same logic as VerifierAgent)
    claims = [c.strip() for c in prediction.split(".") if len(c.strip()) > 5]

    if not claims:
        return 0.0  # nothing hallucinated

    hallucinated = 0

    for claim in claims:
        best_score = 0.0

        for r in retrieved:
            evidence_text = f"{r.get('findings','')} {r.get('impression','')}"
            score = util.cos_sim(
                model.encode(claim),
                model.encode(evidence_text)
            )
            best_score = max(best_score, float(score))

        if best_score < threshold:
            hallucinated += 1

    return hallucinated / len(claims)


# --------------------------
# 4. Evaluate main function
# --------------------------
def evaluate(prediction, retrieved, ground_truth):
    return {
        "semantic_similarity": semantic_similarity(prediction, ground_truth),
        "faithfulness": faithfulness_score(prediction, retrieved),
        "hallucination_rate": hallucination_rate(prediction, retrieved),
        "retrieval_count": len(retrieved)
    }
