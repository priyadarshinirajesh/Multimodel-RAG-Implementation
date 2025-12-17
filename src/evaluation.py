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
    """Checks if model output is grounded in retrieved evidence."""
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
# 3. Evaluate main function
# --------------------------
def evaluate(prediction, retrieved, ground_truth):
    return {
        "semantic_similarity": semantic_similarity(prediction, ground_truth),
        "faithfulness": faithfulness_score(prediction, retrieved),
        "retrieval_count": len(retrieved)
    }
