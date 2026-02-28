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


# Issue 6 FIX: Removed _item_relevance_score and _multimodal_relevance_score.
# Both functions were dead code — never called anywhere in the active metric
# path.  Their presence created maintenance confusion because they used a
# different fusion strategy (weighted average / max) than the inline logic
# inside precision_recall_mrr, making it easy to accidentally use the wrong
# function in future edits.  Deleted entirely.


def _remove_impression(report_text: str) -> str:
    if not report_text:
        return ""
    lower = report_text.lower()
    idx = lower.find("impression:")
    if idx == -1:
        return report_text.strip()
    return report_text[:idx].strip()


def precision_recall_mrr(
    retrieved_docs: list,
    query: str,
    k: int = 5,
    text_threshold: float = 0.28,
    image_threshold: float = 0.24,
) -> dict:
    """
    Query-based multimodal retrieval metrics.

    Relevance is determined by cosine similarity between each evidence unit
    (indication / comparison / findings text, or X-ray image) and the query
    embedding.  This is "pool-based" IR evaluation: no external ground-truth
    set is needed, and Recall@K is measured against the total relevant units
    in the retrieved pool (not an external GT set).

    Precision@K formula  : TP_in_top_K / K   (fixed K denominator — Issue 5)
    Recall@K formula     : TP_in_top_K / total_relevant_in_pool
    MRR                  : 1 / rank_of_first_relevant_hit in top-K

    NOTE for research papers: because Recall uses a pool-based denominator
    (not an external ground-truth set), it should be described as
    "pool-based Recall@K" rather than true Recall@K.
    """
    if not query or not query.strip() or not retrieved_docs:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}

    query_vec = embed_text(query)

    # ── 1. Collect unique text / image units from retrieved evidence ──────────
    indication_set  = set()
    comparison_set  = set()
    findings_set    = set()
    image_set       = set()

    for e in retrieved_docs:
        ind  = (e.get("indication")  or "").strip()
        cmp_ = (e.get("comparison")  or "").strip()
        fnd  = (e.get("findings")    or "").strip()
        img  = e.get("image_path")

        if ind:
            indication_set.add(ind)
        if cmp_:
            comparison_set.add(cmp_)
        if fnd:
            findings_set.add(fnd)
        if img and os.path.exists(img):
            image_set.add(img)

    # ── 2. Build scored unit list ─────────────────────────────────────────────
    units = (
        [("indication", "text",  t) for t in indication_set]
        + [("comparison", "text",  t) for t in comparison_set]
        + [("findings",   "text",  t) for t in findings_set]
        + [("image",      "image", p) for p in image_set]
    )

    if not units:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0}

    # ── 3. Score each unit vs query ───────────────────────────────────────────
    scored = []
    for name, kind, value in units:
        if kind == "text":
            s      = _cosine(query_vec, embed_text(value))
            is_rel = s >= text_threshold
        else:
            s      = _cosine(query_vec, embed_image(value))
            is_rel = s >= image_threshold
        scored.append((name, kind, value, s, is_rel))

    # ── 4. Rank by similarity descending ─────────────────────────────────────
    scored.sort(key=lambda x: x[3], reverse=True)

    # ── 5. Compute metrics ────────────────────────────────────────────────────
    k_eff = min(k, len(scored))
    topk  = scored[:k_eff]

    tp = sum(1 for x in topk if x[4])

    # Issue 5 FIX — Precision@K: always divide by fixed K, not k_eff.
    # Using k_eff (= min(k, pool_size)) inflates precision when fewer than K
    # units exist (denominator shrinks while TP stays the same).
    # Textbook formula: Precision@K = TP_in_top_K / K
    precision = tp / k if k > 0 else 0.0

    # Issue 5 FIX — Recall@K: denominator is ALL relevant units in the FULL
    # scored pool (not just top-k), which is the standard pool-based recall
    # used in IR benchmarks when no separate ground-truth set is available.
    total_rel = sum(1 for x in scored if x[4])
    recall    = tp / total_rel if total_rel > 0 else 0.0

    mrr = 0.0
    for rank, x in enumerate(topk, start=1):
        if x[4]:
            mrr = 1.0 / rank
            break

    return {
        "Precision@K": round(precision, 3),
        "Recall@K":    round(recall,    3),
        "MRR":         round(mrr,       3),
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
    fallback_to_simple: bool = False,
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

        # Issue 10 FIX: guard against NaN / inf from the RAGAS judge.
        # float(nan) succeeds silently; round(nan, 3) returns nan which then
        # propagates into Excel exports as a blank/invalid cell.
        # np.isfinite catches both NaN and ±inf.
        if score is None or not np.isfinite(score):
            return {"score": 0.0, "source": "ragas_failed"}

        return {"score": round(float(score), 3), "source": "ragas"}

    except Exception:
        return {"score": 0.0, "source": "ragas_failed"}