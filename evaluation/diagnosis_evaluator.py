# def role_compliance(answer):
#     violations = [
#         "definitely has",
#         "confirmed diagnosis",
#         "guaranteed",
#         "without doubt"
#     ]

#     for v in violations:
#         if v.lower() in answer.lower():
#             return 0  # Non-compliant

#     return 1  # Compliant

# evaluation/diagnosis_evaluator.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def precision_recall_mrr(retrieved, ground_truth, k=7):
    # Ensure uniqueness
    relevant = set(ground_truth)
    retrieved_k = retrieved[:k]
    retrieved_unique = set(retrieved_k)

    # True positives = unique relevant items retrieved
    tp = len(retrieved_unique & relevant)

    # ✅ FIX: Use actual retrieved count or k, whichever is smaller
    actual_k = min(k, len(retrieved_k))

    precision = tp / actual_k if actual_k > 0 else 0  # ✅ FIXED
    recall = tp / len(relevant) if relevant else 0

    # MRR: first relevant hit
    mrr = 0
    for idx, r in enumerate(retrieved_k, start=1):
        if r in relevant:
            mrr = 1 / idx
            break

    return {
        "Precision@K": round(precision, 3),
        "Recall@K": round(recall, 3),
        "MRR": round(mrr, 3)
    }


def groundedness(answer):
    import re

    lines = [
        l.strip() for l in answer.split("\n")
        if l.strip()
        and not l.lower().startswith("supporting evidence")
        and not l.lower().startswith("next steps")
        and not l.lower().startswith("diagnosis")
    ]

    factual_lines = []
    for l in lines:
        if l.startswith("-") or ":" in l:
            factual_lines.append(l)

    if not factual_lines:
        return 0

    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )

    return round(supported / len(factual_lines), 3)


def clinical_correctness(answer, impressions, threshold=0.55):
    if not impressions:
        return 0

    answer_emb = _embedder.encode([answer])
    imp_embs = _embedder.encode(impressions)

    sims = cosine_similarity(answer_emb, imp_embs)[0]

    return float(max(sims))


def completeness(answer):
    score = 0

    if "[R" in answer:
        score += 1

    if any(w in answer.lower() for w in [
        "recommend", "next step", "follow-up",
        "no further action required", "no further action"
    ]):
        score += 1

    if "diagnosis" in answer.lower():
        score += 1

    return round(score / 3, 2)

