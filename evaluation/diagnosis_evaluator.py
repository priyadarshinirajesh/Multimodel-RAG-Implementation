# evaluation/diagnosis_evaluator.py

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from langchain_groq import ChatGroq

from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image


_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a, b) -> float:
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def _item_relevance_score(e: dict, query_vec, text_w=0.5, image_w=0.5):
    text_score = None
    image_score = None

    text = (e.get("report_text") or "").strip()
    if text:
        text_score = _cosine(query_vec, embed_text(text))

    image_path = e.get("image_path")
    if image_path and os.path.exists(image_path):
        image_score = _cosine(query_vec, embed_image(image_path))

    if text_score is not None and image_score is not None:
        return text_w * text_score + image_w * image_score
    if text_score is not None:
        return text_score
    if image_score is not None:
        return image_score
    return -1.0

def _multimodal_relevance_score(evidence_item: dict, query_vec) -> float:
    text_score = None
    image_score = None

    report_text = (evidence_item.get("report_text") or "").strip()
    image_path = evidence_item.get("image_path")

    # Text relevance
    if report_text:
        try:
            text_vec = embed_text(report_text)
            text_score = _cosine(query_vec, text_vec)
        except Exception:
            text_score = None

    # Image relevance
    if image_path and os.path.exists(image_path):
        try:
            img_vec = embed_image(image_path)
            image_score = _cosine(query_vec, img_vec)
        except Exception:
            image_score = None

    # Fusion rule
    if text_score is not None and image_score is not None:
        return max(text_score, image_score)  # strict "either modality relevant"
    if text_score is not None:
        return text_score
    if image_score is not None:
        return image_score
    return -1.0



def _remove_impression(report_text: str) -> str:
    if not report_text:
        return ""
    lower = report_text.lower()
    idx = lower.find("impression:")
    if idx == -1:
        return report_text.strip()
    return report_text[:idx].strip()


# def precision_recall_mrr(retrieved_docs: list, query: str, k: int = 7, relevance_threshold: float = 0.28) -> dict:
#     """
#     Query-based multimodal retrieval metrics.
#     A retrieved item is relevant if fused(text,image) similarity with query >= threshold.
#     """
#     if not query or not query.strip():
#         return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}
#     if not retrieved_docs:
#         return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}

#     query_vec = embed_text(query)
#     top_k_docs = retrieved_docs[:k]

#     topk_scores = [_multimodal_relevance_score(e, query_vec) for e in top_k_docs]
#     all_scores = [_multimodal_relevance_score(e, query_vec) for e in retrieved_docs]

#     topk_relevant = [s >= relevance_threshold for s in topk_scores]
#     all_relevant = [s >= relevance_threshold for s in all_scores]

#     tp_at_k = sum(topk_relevant)
#     precision = tp_at_k / k if k > 0 else 0.0

#     total_relevant = sum(all_relevant)
#     recall = tp_at_k / total_relevant if total_relevant > 0 else 0.0

#     mrr = 0.0
#     for rank, is_rel in enumerate(topk_relevant, start=1):
#         if is_rel:
#             mrr = 1.0 / rank
#             break

#     return {
#         "Precision@K": round(precision, 3),
#         "Recall@K": round(recall, 3),
#         "MRR": round(mrr, 3),
#     }

def precision_recall_mrr(
    retrieved_docs: list,
    query: str,
    k: int = 5,
    text_threshold: float = 0.28,
    image_threshold: float = 0.24,
):
    if not query or not query.strip() or not retrieved_docs:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}

    query_vec = embed_text(query)

    # 1) Collect unique text units in sets
    indication_set = set()
    comparison_set = set()
    findings_set = set()

    image_set = set()

    for e in retrieved_docs:
        ind = (e.get("indication") or "").strip()
        cmp_ = (e.get("comparison") or "").strip()
        fnd = (e.get("findings") or "").strip()

        if ind:
            indication_set.add(ind)
        if cmp_:
            comparison_set.add(cmp_)
        if fnd:
            findings_set.add(fnd)

        img = e.get("image_path")
        if img and os.path.exists(img):
            image_set.add(img)

    # 2) Convert sets -> lists (as you requested)
    indication_list = list(indication_set)
    comparison_list = list(comparison_set)
    findings_list = list(findings_set)
    image_list = list(image_set)

    # 3) Build unit list (dynamic size)
    units = []
    for t in indication_list:
        units.append(("indication", "text", t))
    for t in comparison_list:
        units.append(("comparison", "text", t))
    for t in findings_list:
        units.append(("findings", "text", t))
    for p in image_list:
        units.append(("image", "image", p))

    if not units:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}


    # 4) Score each unit vs query
    scored = []
    for name, kind, value in units:
        if kind == "text":
            s = _cosine(query_vec, embed_text(value))
            is_rel = s >= text_threshold
        else:
            s = _cosine(query_vec, embed_image(value))
            is_rel = s >= image_threshold
        scored.append((name, kind, value, s, is_rel))

    # 5) Rank by similarity descending
    scored.sort(key=lambda x: x[3], reverse=True)

    # 6) Dynamic K clamp (important)
    k_eff = min(k, len(scored))
    topk = scored[:k_eff]

    tp = sum(1 for x in topk if x[4])
    precision = tp / k_eff if k_eff > 0 else 0.0

    total_rel = sum(1 for x in scored if x[4])
    recall = tp / total_rel if total_rel > 0 else 0.0

    mrr = 0.0
    for rank, x in enumerate(topk, start=1):
        if x[4]:
            mrr = 1.0 / rank
            break

    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "MRR": mrr,
    }



def groundedness(answer: str) -> float:
    lines = [
        l.strip() for l in answer.split("\n")
        if l.strip()
        and not l.lower().startswith("supporting evidence")
        and not l.lower().startswith("next steps")
        and not l.lower().startswith("diagnosis")
    ]
    factual_lines = [l for l in lines if l.startswith("-") or ":" in l]
    if not factual_lines:
        return 0.0
    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )
    return round(supported / len(factual_lines), 3)


def groundedness_simple(answer: str) -> float:
    lines = [
        l.strip() for l in answer.split("\n")
        if l.strip()
        and not l.lower().startswith("supporting evidence")
        and not l.lower().startswith("next steps")
        and not l.lower().startswith("diagnosis")
    ]
    factual_lines = [l for l in lines if l.startswith("-") or ":" in l]
    if not factual_lines:
        return 0.0
    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )
    return round(supported / len(factual_lines), 3)


def clinical_correctness(answer: str, impressions: list, threshold: float = 0.55) -> float:
    if not impressions:
        return 0.0
    answer_emb = _embedder.encode([answer])
    imp_embs   = _embedder.encode(impressions)
    sims       = cosine_similarity(answer_emb, imp_embs)[0]
    return float(max(sims))


def completeness(answer: str) -> float:
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


def _build_ragas_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing for RAGAS groundedness.")
    model_name = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model_name, api_key=api_key, temperature=0)


def _extract_contexts(evidence: list) -> list:
    contexts = []
    for e in evidence:
        txt = _remove_impression(e.get("report_text", ""))
        if txt and txt.strip():
            contexts.append(txt.strip())
    return contexts


def groundedness_ragas(
    query: str,
    answer: str,
    evidence: list,
    fallback_to_simple: bool = False
) -> dict:
    contexts = _extract_contexts(evidence)
    if not query or not answer or not contexts:
        return {"score": 0.0, "source": "ragas_failed"}

    try:
        dataset = Dataset.from_dict({
            "question": [query],
            "answer":   [answer],
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