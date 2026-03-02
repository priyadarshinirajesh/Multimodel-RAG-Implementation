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
    Clinical reasoning agent — upgraded to Clinical Assistant level.

    Produces four-section structured responses:
      - Clinical Impression  (present / absent / indeterminate + confidence)
      - Evidence Synthesis   (unique observations, discordance notes)
      - Differential Considerations (primary + alternative diagnoses)
      - Actionable Next Steps (specific actions, not generic filler)

    Args:
        query:     Clinical query string.
        evidence:  Filtered, ranked evidence dicts (relevance_score desc).
        user_role: Kept for backward compatibility / RBAC logging.
    """

    logger.info("Starting clinical reasoning")
    logger.info(f"Evidence items received: {len(evidence)}")

    # Used by ClinicalCorrectness: does the LLM answer agree with retrieved impressions?
    ground_truth_impressions = [e["report_text"] for e in evidence]

    # Image analysis via LLaVA-Med
    image_insights = image_insight_agent_llava_med(evidence, query)

    # ── Pathology findings from DenseNet (added by evidence_aggregation_agent) ──
    pathology_findings = []
    for idx, e in enumerate(evidence, start=1):
        if "pathology_findings" in e and e["pathology_findings"]:
            pf = e["pathology_findings"]
            # Only include if something meaningful was detected
            if "No significant pathologies" not in pf and "No image available" not in pf:
                pathology_findings.append(f"[R{idx}] {pf}")

    # ── Build combined evidence list ──────────────────────────────────────────
    combined_evidence = []
    for i, e in enumerate(evidence,start=1):
        report = _remove_impression(e.get('report_text',''))
        if report.strip():
            combined_evidence.append(f"[R{i}] ({e['modality']}) {report}")
        else:
            combined_evidence.append(
                f"[R{i}] ({e['modality']}) [NO TEXT REPORT AVAILABLE — imaging only]"
            )
    combined_evidence.extend(image_insights)

    # ── Check if text report evidence exists (for confidence indicator logic) ─
    has_text_reports = any(
        e.get("report_text", "").strip()
        for e in evidence
    )

    # ── Build pathology summary for discordance detection ─────────────────────
    pathology_summary = []
    for idx, e in enumerate(evidence, start=1):
        top = e.get("top_pathologies", [])
        if top:
            items = ", ".join([f"{p}: {s*100:.1f}%" for p, s in top])
            pathology_summary.append(f"[R{idx}] CNN detected: {items}")

    system_prompt = (
        "You are a senior clinical decision-support AI assisting medical professionals. "
        "Your job is to synthesize multi-source evidence into actionable clinical intelligence. "
        "You clearly distinguish between text report evidence and AI image findings, "
        "flag discordances between CNN results and text reports, "
        "and always provide specific, clinically meaningful next steps — never generic filler."
    )

    prompt = f"""
You are a clinical decision-support AI.

========================================
PATHOLOGY DETECTION RESULTS (DenseNet CNN — AI model, not a radiologist):
========================================
{chr(10).join(pathology_summary) if pathology_summary else "No pathologies detected above threshold by CNN."}

========================================
ABSOLUTE RULES (MANDATORY — NEVER BREAK THESE):
========================================
1. Use ONLY the evidence provided below. DO NOT infer or hallucinate beyond evidence.
2. EVERY factual sentence MUST end with a citation: [R1], [R2], [R1-IMAGE], etc.
3. If evidence is insufficient for a finding → write: "Insufficient evidence to determine [X]. [Rx]"
4. Do NOT repeat the same finding in different words across bullets — merge duplicates into one.
5. PHANTOM CITATION RULE (CRITICAL):
   Evidence items marked [NO TEXT REPORT AVAILABLE] have NO text data.
   You MUST NOT use [R#] (text citation) for any such item.
   You MAY only use [R#-IMAGE] if an image insight exists for that item.
   If neither text nor image insight exists → do not cite that evidence item at all.
6. CLINICAL IMPRESSION RULE: The Clinical Impression must answer the clinical question, NOT restate patient symptoms.
   
========================================
EVIDENCE HIERARCHY (FOLLOW STRICTLY):
========================================
- TEXT REPORTS [R1], [R2]  →  PRIMARY source. Base Clinical Impression primarily on these.
- IMAGE INSIGHTS [R1-IMAGE] →  SECONDARY source. Use to support text, NEVER replace it.
- CNN PATHOLOGY SCORES      →  SUPPLEMENTARY. Cross-check against text reports.

If a finding is supported ONLY by image/CNN with NO text report backing:
→ Begin that sentence with: "Based on imaging only (limited confidence):"

If text report AND CNN findings CONTRADICT each other:
→ You MUST write a "Discordance Note:" bullet in Evidence Synthesis.

========================================
DISCORDANCE DETECTION RULE:
========================================
Compare the CNN pathology results above against the text reports in the evidence below.
- If CNN detected a pathology above 40% confidence BUT text report does NOT mention it → flag it.
- If text report mentions a finding that CNN did NOT detect → flag it.
- Write discordance as: "Discordance Note: CNN detected [X] at [Y]% but text report states [Z]. [R1, R1-IMAGE]"

========================================
ANTI-REPETITION RULE:
========================================
Each bullet in Evidence Synthesis must make a UNIQUE clinical observation.
If two bullets say the same thing in different words → merge them into one.
Minimum 2, maximum 4 bullets in Evidence Synthesis.

========================================
BANNED RECOMMENDATION PHRASES (NEVER USE THESE):
========================================
- "Further evaluation by a healthcare professional is necessary"
- "Consult a doctor"
- "Seek medical advice"
- "Clinical correlation is recommended" ← only allowed if followed by WHAT to correlate with

REQUIRED recommendation format:
[Specific action] because [clinical reason from findings] [timeframe if relevant]. [Rx]

GOOD recommendation examples:
- "Lateral chest X-ray recommended to confirm right middle lobe infiltrate suggested by CNN findings. [Rx]"
- "Compare with prior chest films from [date] to assess interval change in cardiac silhouette size. [Rx]"
- "ECG and echocardiogram indicated given borderline cardiothoracic ratio noted on imaging. [Rx]"
- "Repeat PA chest X-ray in 48 hours if respiratory symptoms persist. [Rx]"

========================================
CONFIDENCE INDICATOR (ADD TO CLINICAL IMPRESSION):
========================================
End your Clinical Impression sentence with ONE of:
- [HIGH CONFIDENCE — supported by text report]
- [MODERATE CONFIDENCE — partial text support, supplemented by imaging]  
- [LOW CONFIDENCE — imaging/CNN only, no text report support]

{"NOTE: Text report evidence IS available — base your impression on it." if has_text_reports else "WARNING: Text report evidence is LIMITED or absent — rely on imaging with low confidence."}

========================================
Retrieved Clinical Evidence:
========================================
{chr(10).join(combined_evidence)}

========================================
Clinical Question:
========================================
{query}

========================================
RESPONSE FORMAT — FOLLOW THIS EXACTLY:
========================================

Clinical Impression:
[One concise sentence answering the clinical question directly: 
 Is active disease / the queried pathology PRESENT, ABSENT, or INDETERMINATE on imaging?
 Do NOT restate symptoms from the query. State the RADIOLOGICAL conclusion.
 End with confidence indicator.]

Evidence Synthesis:
- [Unique observation from text report]. [R1]
- [Unique observation from imaging]. [R1-IMAGE]
- [Discordance Note if applicable]. [R1, R1-IMAGE]

Differential Considerations:
- Primary: [Most likely diagnosis based on evidence and why]. [R1]
- Alternative: [What else it could be and why evidence points away from it]. [Rx or R1 — only cite evidence that EXISTS]

========================================
NOW RESPOND IN THE EXACT FORMAT ABOVE. DO NOT ADD ANY OTHER SECTIONS.
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


    print("=======FINAL ANSWER=========")
    print(final_answer)
    print("============================")

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    metrics = {}

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