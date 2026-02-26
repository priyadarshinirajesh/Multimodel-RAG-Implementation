# evaluation/diagnosis_evaluator.py

import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from langchain_groq import ChatGroq


_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _remove_impression(report_text: str) -> str:
    if not report_text:
        return ""
    lower = report_text.lower()
    idx = lower.find("impression:")
    if idx == -1:
        return report_text.strip()
    return report_text[:idx].strip()

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

def groundedness_simple(answer: str) -> float:
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
        return 0.0

    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )
    return round(supported / len(factual_lines), 3)


def _build_ragas_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing for RAGAS groundedness.")
    model_name = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model_name, api_key=api_key, temperature=0)


def _extract_contexts(evidence: list) -> list[str]:
    contexts = []
    for e in evidence:
        txt = _remove_impression(e.get("report_text", ""))
        if txt and txt.strip():
            contexts.append(txt.strip())
    return contexts



# def groundedness_ragas(query: str, answer: str, evidence: list, fallback_to_simple: bool = True) -> dict:
#     contexts = _extract_contexts(evidence)
#     if not query or not answer or not contexts:
#         fallback = groundedness_simple(answer) if fallback_to_simple else 0.0
#         return {"score": fallback, "source": "fallback"}

#     try:
#         dataset = Dataset.from_dict({
#             "question": [query],
#             "answer": [answer],
#             "contexts": [contexts],
#         })

#         result = evaluate(
#             dataset=dataset,
#             metrics=[Faithfulness()],
#             llm=_build_ragas_llm(),
#         )

#         score = None
#         try:
#             df = result.to_pandas()
#             if "faithfulness" in df.columns:
#                 score = float(df.iloc[0]["faithfulness"])
#         except Exception:
#             pass

#         if score is None:
#             try:
#                 score = float(result["faithfulness"])
#             except Exception:
#                 pass

#         if score is None:
#             fallback = groundedness_simple(answer) if fallback_to_simple else 0.0
#             return {"score": fallback, "source": "fallback"}

#         return {"score": round(score, 3), "source": "ragas"}

#     except Exception:
#         fallback = groundedness_simple(answer) if fallback_to_simple else 0.0
#         return {"score": fallback, "source": "fallback"}

def groundedness_ragas(query: str, answer: str, evidence: list, fallback_to_simple: bool = False) -> dict:
    contexts = _extract_contexts(evidence)

    if not query or not answer or not contexts:
        return {"score": 0.0, "source": "ragas_failed"}

    try:
        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness()],
            llm=_build_ragas_llm(),
        )

        score = None
        try:
            df = result.to_pandas()
            if "faithfulness" in df.columns:
                score = float(df.iloc[0]["faithfulness"])
        except Exception:
            pass

        if score is None:
            try:
                score = float(result["faithfulness"])
            except Exception:
                pass

        if score is None:
            return {"score": 0.0, "source": "ragas_failed"}

        return {"score": round(score, 3), "source": "ragas"}

    except Exception:
        return {"score": 0.0, "source": "ragas_failed"}
