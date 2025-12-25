# evaluation/diagnosis_evaluator.py

def precision_recall_mrr(retrieved, ground_truth, k=7):
    relevant = set(ground_truth)

    retrieved_k = retrieved[:k]
    retrieved_set = set(retrieved_k)

    precision = len(retrieved_set & relevant) / k if k else 0
    recall = len(retrieved_set & relevant) / len(relevant) if relevant else 0

    mrr = 0
    for idx, item in enumerate(retrieved_k, start=1):
        if item in relevant:
            mrr = 1 / idx
            break

    return {
        "Precision@K": round(precision, 3),
        "Recall@K": round(recall, 3),
        "MRR": round(mrr, 3)
    }

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

import re

def groundedness(answer):
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', answer)
        if s.strip()
    ]

    supported = 0
    for s in sentences:
        if re.search(r"\[R\d+(-IMAGE)?\]", s):
            supported += 1

    return round(supported / len(sentences), 3) if sentences else 0


def clinical_correctness(answer, impressions):
    for imp in impressions:
        key_terms = imp.lower().split()[:5]  # simple heuristic
        if any(term in answer.lower() for term in key_terms):
            return 1
    return 0

def completeness(answer):
    score = 0

    if any(w in answer.lower() for w in ["diagnosis", "condition", "suggests"]):
        score += 1

    if "[R" in answer:
        score += 1

    if any(w in answer.lower() for w in ["recommend", "next step", "follow-up"]):
        score += 1

    return round(score / 3, 2)




