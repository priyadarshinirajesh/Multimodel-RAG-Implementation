# agents/clinical_reasoning_agent.py

import os
import requests
from dotenv import load_dotenv

from evaluation.diagnosis_evaluator import (
    precision_recall_mrr,
    groundedness,
    clinical_correctness,
    completeness,
    groundedness_ragas,
    groundedness_simple,
)

from agents.image_insight_agent_llava_med import image_insight_agent_llava_med
from agents.verifiers.structure_repair import enforce_structure
from utils.logger import get_logger

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found")

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

logger = get_logger("ReasoningAgent")


def _remove_impression(report_text: str) -> str:
    if not report_text:
        return ""
    lower = report_text.lower()
    idx = lower.find("impression:")
    if idx == -1:
        return report_text.strip()
    return report_text[:idx].strip()


def _extract_impression(report_text: str) -> str:
    if not report_text:
        return ""
    lower = report_text.lower()
    idx = lower.find("impression:")
    if idx == -1:
        return ""
    return report_text[idx + len("impression:"):].strip()


def clinical_reasoning_agent(
    query: str,
    evidence: list,
    user_role: str = "doctor",
):
    """
    Clinical reasoning agent.

    Args:
        query:     Clinical query string.
        evidence:  Filtered, ranked evidence dicts (relevance_score desc).
                   Each dict must have keys: report_text, modality, image_path (optional),
                   indication, comparison, findings (used by precision_recall_mrr).
        user_role: Kept for backward compatibility / RBAC logging.

    Retrieval metrics (Precision@K, Recall@K, MRR):
        Computed via precision_recall_mrr() using a QUERY-BASED approach.
        Each evidence unit (indication / comparison / findings text, or image)
        is scored by cosine similarity to the query embedding.
        A unit is "relevant" if its similarity meets the configured threshold.
        This is pool-based IR evaluation — no external ground-truth set is
        needed, and Recall@K is measured against the relevant units in the
        retrieved pool.

    Groundedness:
        Computed via RAGAS Faithfulness (LLM-as-judge).  Falls back to
        citation-counting heuristic if RAGAS fails.

    ClinicalCorrectness:
        Semantic similarity between generated answer and report impressions
        from the retrieved evidence.  Measures whether the answer agrees
        with what the retrieved records actually say.
    """

    logger.info("Starting clinical reasoning")
    logger.info(f"Evidence items received: {len(evidence)}")

    # Used by ClinicalCorrectness: does the LLM answer agree with retrieved impressions?
    ground_truth_impressions = [e["report_text"] for e in evidence]

    # Image analysis via LLaVA-Med
    image_insights = image_insight_agent_llava_med(evidence, query)

    # Pathology findings from DenseNet (added by evidence_aggregation_agent)
    pathology_findings = []
    for idx, e in enumerate(evidence, start=1):
        if "pathology_findings" in e and e["pathology_findings"]:
            pathology_findings.append(f"[R{idx}] {e['pathology_findings']}")

    combined_evidence = [
        f"[R{i}] ({e['modality']}) {_remove_impression(e.get('report_text', ''))}"
        for i, e in enumerate(evidence, start=1)
    ]
    combined_evidence.extend(image_insights)

    system_prompt = (
        "You are a clinical decision-support AI for medical professionals. "
        "Provide detailed, evidence-based clinical reasoning."
    )

    prompt = f"""
You are a clinical decision-support AI.

PATHOLOGY DETECTION RESULTS (from DenseNet CNN):
{chr(10).join(pathology_findings) if pathology_findings else "No pathologies detected above threshold"}

ABSOLUTE RULES (MANDATORY):
- Use ONLY the evidence provided below.
- DO NOT infer, assume, or diagnose beyond evidence.
- EVERY factual sentence MUST end with a citation like [R1], [R2], etc.
- If evidence is insufficient, explicitly write: "Insufficient evidence [Rx]".

EVIDENCE USAGE RULES:
- Prefer text reports over image descriptions when both exist.
- Image insights are SUPPORTING only, not primary diagnostic proof.
- Ignore modalities that are clinically irrelevant to the question.

Retrieved Clinical Evidence:
{chr(10).join(combined_evidence)}

Clinical Question:
{query}

========================================
RESPONSE FORMAT (YOU MUST FOLLOW THIS):
========================================

Diagnosis / Impression:
[Write ONE concise sentence with citation, e.g., "- No acute abnormality detected. [R1]"]

Supporting Evidence:
[Write 2-4 bullet points with citations, e.g.:
- Finding 1 description. [R1]
- Finding 2 description. [R2]]

Next Steps / Recommendations:
[Write 1-2 bullet points with citations or [Rx], e.g.:
- Clinical correlation recommended. [Rx]
- Follow-up imaging if symptoms persist. [Rx]]

========================================
NOW RESPOND IN THE EXACT FORMAT ABOVE:
========================================
"""

    print("=======FINAL PROMPT=========")
    print(prompt)
    print("============================")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()

    raw_answer   = response.json()["choices"][0]["message"]["content"]
    final_answer = enforce_structure(raw_answer)

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    metrics = {}

    # Query-based retrieval metrics.
    # precision_recall_mrr scores each evidence unit (indication / comparison /
    # findings / image) by cosine similarity to the query.  A unit is "relevant"
    # if its similarity >= text_threshold (text) or image_threshold (image).
    # See diagnosis_evaluator.py for formula details and research-paper caveats.
    metrics.update(
        precision_recall_mrr(
            retrieved_docs=evidence,
            query=query,
            k=5,
            text_threshold=0.30,
            image_threshold=0.30,
        )
    )

    g = groundedness_ragas(
        query=query,
        answer=final_answer,
        evidence=evidence,
        fallback_to_simple=False,
    )
    metrics["Groundedness"]       = g["score"]
    metrics["GroundednessSource"] = g["source"]
    metrics["GroundednessSimple"] = groundedness_simple(final_answer)

    metrics["ClinicalCorrectness"] = clinical_correctness(
        final_answer, ground_truth_impressions
    )
    metrics["Completeness"] = completeness(final_answer)

    return {
        "final_answer": final_answer,
        "metrics":      metrics,
    }