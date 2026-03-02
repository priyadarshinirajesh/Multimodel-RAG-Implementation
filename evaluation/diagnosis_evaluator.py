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

    Precision@K formula  : TP_in_top_K / K   (fixed K denominator)
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
    indication_set = set()
    comparison_set = set()
    findings_set   = set()
    image_set      = set()

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

    # Precision@K: always divide by fixed K, not k_eff.
    # Using k_eff inflates precision when fewer than K units exist.
    # Textbook formula: Precision@K = TP_in_top_K / K
    precision = tp / k if k > 0 else 0.0

    # Recall@K: denominator is ALL relevant units in the FULL scored pool
    # (not just top-k) — standard pool-based recall used in IR benchmarks
    # when no separate ground-truth set is available.
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
    """
    Simple citation-based groundedness for the new four-section format.
    Counts factual lines that carry a [R#] or [R#-IMAGE] citation.
    Section header lines are excluded from the count.
    """
    lines = [
        l.strip() for l in answer.split("\n")
        if l.strip()
        and not l.lower().startswith("clinical impression")
        and not l.lower().startswith("evidence synthesis")
        and not l.lower().startswith("differential consideration")
        # emoji-prefixed section headers
        and not l.startswith("🩺")
        and not l.startswith("🔍")
        and not l.startswith("🔀")
        and not l.startswith("📋")
    ]

    # Factual lines: bullets, key:value lines, or differential markers
    factual_lines = [
        l for l in lines
        if l.startswith("-")
        or ":" in l
        or l.startswith("✅")
        or l.startswith("🔄")
    ]

    if not factual_lines:
        return 0.0

    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )
    return round(supported / len(factual_lines), 3)


def groundedness_simple(answer: str) -> float:
    """
    Alias of groundedness() — used as fallback when RAGAS judge is unavailable.
    Kept as a separate function so call sites don't need to change.
    """
    return groundedness(answer)


# def clinical_correctness(answer: str, impressions: list, threshold: float = 0.55) -> float:
#     if not impressions:
#         return 0.0
    
#     print("======ANSWER FOR CLINICAL CORRECTNESS======")
#     print(answer)
#     print("===========================================")

#     new_imp = impressions[0].split("Impression:")
#     new_imp = new_imp[-1]
#     print("======IMPRESSIONS FOR CLINICAL CORRECTNESS======")
#     print(new_imp)
#     print("===============================================")


#     answer_emb = _embedder.encode([answer])
#     imp_embs   = _embedder.encode(new_imp)
#     sims       = cosine_similarity(answer_emb, imp_embs)[0]
#     return float(max(sims))

def clinical_correctness(answer: str, impressions: list, threshold: float = 0.55) -> float:
    if not impressions:
        return 0.0

    cleaned = []
    for imp in impressions:
        txt = (imp or "").strip()
        if not txt:
            continue
        parts = txt.split("Impression:")
        cleaned_txt = parts[-1].strip() if len(parts) > 1 else txt
        if cleaned_txt:
            cleaned.append(cleaned_txt)

    if not cleaned:
        return 0.0
    
    print("======ANSWER FOR CLINICAL CORRECTNESS======")
    new_answer = answer.split("Evidence Synthesis:")[0]
    new_answer = new_answer.split("Clinical Impression:")[-1]
    new_answer = new_answer.split("[")[0]
    print(new_answer)
    print("===========================================")

    print("======IMPRESSIONS FOR CLINICAL CORRECTNESS======")
    print(cleaned)
    print("===============================================")
    

    answer_emb = _embedder.encode([new_answer])     # shape (1, d)
    imp_embs = _embedder.encode(cleaned)        # shape (n, d)
    sims = cosine_similarity(answer_emb, imp_embs)[0]
    return float(max(sims))


def completeness(answer: str) -> float:
    """
    Five-point completeness check aligned with the new four-section
    clinical response format:

      1. Clinical Impression section present
      2. Evidence Synthesis section present AND contains ≥1 citation
      3. Differential Considerations section present with a Primary diagnosis
      4. Confidence indicator (HIGH / MODERATE / LOW CONFIDENCE) present
    """
    score = 0
    lower = answer.lower()

    # 1. Clinical Impression section present
    if "clinical impression" in lower:
        score += 1

    # 2. Evidence Synthesis with at least one [R#] or [R#-IMAGE] citation
    if "evidence synthesis" in lower and re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", answer):
        score += 1

    # 3. Differential Considerations with a Primary diagnosis marker
    if "differential consideration" in lower and (
        "primary:" in lower
        or "✅" in answer
    ):
        score += 1

    # 4. Confidence indicator present
    if re.search(r"\b(HIGH|MODERATE|LOW)\s+CONFIDENCE\b", answer, re.IGNORECASE):
        score += 1


    return round(score / 4, 2)


def _build_ragas_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing for RAGAS groundedness.")
    model_name = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model_name, api_key=api_key, temperature=0)


def _extract_contexts(evidence: list) -> list:
    """
    Extract report text from evidence items as RAGAS context strings.
    Strips the radiologist impression from each report so the LLM judge
    evaluates faithfulness against findings only (not the gold-standard
    conclusion), avoiding circular scoring.
    """
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
    """
    LLM-as-judge groundedness via RAGAS Faithfulness metric.

    Sends (query, answer, contexts) to a Groq-hosted LLM judge.
    Falls back to groundedness_simple() if RAGAS fails or returns NaN/inf.

    Returns:
        {"score": float, "source": "ragas" | "ragas_failed"}
    """
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

        # Guard against NaN / inf from the RAGAS judge.
        # float(nan) succeeds silently; np.isfinite catches both NaN and ±inf.
        if score is None or not np.isfinite(score):
            return {"score": 0.0, "source": "ragas_failed"}

        return {"score": round(float(score), 3), "source": "ragas"}

    except Exception:
        return {"score": 0.0, "source": "ragas_failed"}