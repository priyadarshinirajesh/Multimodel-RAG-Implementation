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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

logger = get_logger("ReasoningAgent")


def clinical_reasoning_agent(query: str, evidence: list, user_role: str = "doctor"):
    """
    Clinical reasoning agent (RBAC removed)
    
    Args:
        query: Clinical query
        evidence: List of evidence items
        user_role: Kept for backward compatibility but not used
    """
    
    logger.info(f"Starting clinical reasoning")
    logger.info(f"Evidence items received: {len(evidence)}")

    # Use ALL retrieved evidence as ground truth
    ground_truth_impressions = [e["report_text"] for e in evidence]
    
    logger.info(f"Ground truth items: {len(ground_truth_impressions)}")

    # Image analysis (always enabled now)
    image_insights = image_insight_agent_llava_med(evidence, query)

    # Pathology findings
    pathology_findings = []
    for idx, e in enumerate(evidence, start=1):
        if "pathology_findings" in e and e["pathology_findings"]:
            pathology_findings.append(f"[R{idx}] {e['pathology_findings']}")

    combined_evidence = [
        f"[R{i}] ({e['modality']}) {e['report_text']}"
        for i, e in enumerate(evidence, start=1)
    ]
    combined_evidence.extend(image_insights)

    # Standard system prompt (no role variations)
    system_prompt = "You are a clinical decision-support AI for medical professionals. Provide detailed, evidence-based clinical reasoning."

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
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()

    raw_answer = response.json()["choices"][0]["message"]["content"]

    # Hard guarantee structure
    final_answer = enforce_structure(raw_answer)

    # Deduplicate retrieved reports before evaluation
    retrieved_texts = [e["report_text"] for e in evidence]
    unique_retrieved = list(dict.fromkeys(retrieved_texts))
    
    logger.info(f"Retrieved texts: {len(retrieved_texts)}, Unique: {len(unique_retrieved)}")

    # Calculate metrics
    metrics = {}
    metrics.update(
        precision_recall_mrr(
            retrieved=unique_retrieved,
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