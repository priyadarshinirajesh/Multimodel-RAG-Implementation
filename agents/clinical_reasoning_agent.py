# agents/clinical_reasoning_agent.py

import os
import requests
from dotenv import load_dotenv

from evaluation.diagnosis_evaluator import (
    precision_recall_mrr,
    groundedness,
    clinical_correctness,
    completeness
)

from agents.image_insight_agent_llava_med import image_insight_agent_llava_med
from agents.verifiers.structure_repair import enforce_structure
from utils.logger import get_logger

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

logger = get_logger("ReasoningAgent")


def clinical_reasoning_agent(query: str, evidence: list):
    logger.info("Starting clinical reasoning")
    logger.info(f"Evidence items received: {len(evidence)}")

    ground_truth_impressions = [
        e["report_text"]
        for e in evidence
        if "impression" in e["report_text"].lower()
    ]

    image_insights = image_insight_agent_llava_med(evidence, query)

    combined_evidence = [
        f"[R{i}] ({e['modality']}) {e['report_text']}"
        for i, e in enumerate(evidence, start=1)
    ]
    combined_evidence.extend(image_insights)

    prompt = f"""
You are a clinical decision-support AI.

ABSOLUTE RULES (MANDATORY):
- Use ONLY the evidence provided below.
- DO NOT infer, assume, or diagnose beyond evidence.
- EVERY factual sentence MUST end with a citation.
- If evidence is insufficient, explicitly write: "Insufficient evidence [Rx]".

EVIDENCE USAGE RULES:
- Prefer text reports over image descriptions when both exist.
- Image insights are SUPPORTING only, not primary diagnostic proof.
- Ignore modalities that are clinically irrelevant to the question.

Retrieved Clinical Evidence:
{chr(10).join(combined_evidence)}

Clinical Question:
{query}

Respond EXACTLY in this structure:

Diagnosis / Impression:
- One short sentence with citation

Supporting Evidence:
- 2–4 bullets with citations

Next Steps / Recommendations:
- 1–2 bullets with [Rx]
"""
    print("=== Prompt ===")
    print(prompt)
    print("==============")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You generate strict clinical reports."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()

    raw_answer = response.json()["choices"][0]["message"]["content"]

    # 🔒 HARD GUARANTEE
    final_answer = enforce_structure(raw_answer)

    metrics = {}
    metrics.update(
        precision_recall_mrr(
            retrieved=[e["report_text"] for e in evidence],
            ground_truth=ground_truth_impressions,
            k=7
        )
    )

    metrics["Groundedness"] = groundedness(final_answer)
    metrics["ClinicalCorrectness"] = clinical_correctness(
        final_answer, ground_truth_impressions
    )
    metrics["Completeness"] = completeness(final_answer)

    return {
        "final_answer": final_answer,
        "metrics": metrics
    }
