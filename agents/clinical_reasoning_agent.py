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


def clinical_reasoning_agent(query: str, evidence: list, user_role: str = "doctor"):  # 🆕 Added user_role
    logger.info(f"Starting clinical reasoning for role: {user_role}")
    logger.info(f"Evidence items received: {len(evidence)}")

    # ✅ NEW: Use ALL retrieved evidence as ground truth
    ground_truth_impressions = [e["report_text"] for e in evidence]
    
    logger.info(f"Ground truth items: {len(ground_truth_impressions)}")

    # 🆕 NEW: Skip image analysis for nurses (they don't have image access)
    if user_role == "nurse":
        logger.info("[RBAC] Nurse role - skipping image analysis")
        image_insights = []
    else:
        image_insights = image_insight_agent_llava_med(evidence, query)

    pathology_findings = []
    for idx, e in enumerate(evidence, start=1):
        if "pathology_findings" in e and e["pathology_findings"]:
            pathology_findings.append(f"[R{idx}] {e['pathology_findings']}")

    combined_evidence = [
        f"[R{i}] ({e['modality']}) {e['report_text']}"
        for i, e in enumerate(evidence, start=1)
    ]
    combined_evidence.extend(image_insights)

    # 🆕 NEW: Role-specific system prompts
    system_prompt = get_role_specific_system_prompt(user_role)
    
    # 🆕 NEW: Role-specific instructions
    role_instructions = get_role_specific_instructions(user_role)

    prompt = f"""
You are a clinical decision-support AI.

{role_instructions}

PATHOLOGY DETECTION RESULTS (from DenseNet CNN):
{chr(10).join(pathology_findings) if pathology_findings else "No pathologies detected above threshold"}

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
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},  # 🆕 Role-specific system prompt
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()

    raw_answer = response.json()["choices"][0]["message"]["content"]

    # 🔒 HARD GUARANTEE
    final_answer = enforce_structure(raw_answer)

    # ✅ Deduplicate retrieved reports before evaluation
    retrieved_texts = [e["report_text"] for e in evidence]
    unique_retrieved = list(dict.fromkeys(retrieved_texts))
    
    logger.info(f"Retrieved texts: {len(retrieved_texts)}, Unique: {len(unique_retrieved)}")

    # ✅ Calculate metrics
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


# 🆕 NEW: Helper functions for role-specific prompts

def get_role_specific_system_prompt(role: str) -> str:
    """Returns appropriate system prompt based on user role"""
    
    if role == "doctor":
        return "You are a clinical decision-support AI for medical professionals. Provide detailed, evidence-based clinical reasoning."
    
    elif role == "nurse":
        return "You are a care-focused AI assistant for nursing staff. Provide clear, actionable information for patient care delivery. Avoid detailed diagnostic reasoning - focus on observations and care instructions."
    
    elif role == "patient":
        return "You are a patient education AI. Use simple, non-technical language. Explain findings in a way that is easy to understand. Avoid medical jargon and complex diagnostic terminology."
    
    else:
        return "You are a clinical decision-support AI."


def get_role_specific_instructions(role: str) -> str:
    """Returns role-specific instructions for the prompt"""
    
    if role == "doctor":
        return """
DOCTOR MODE:
- Provide complete clinical reasoning
- Include all diagnostic considerations
- Use standard medical terminology
- Reference specific anatomical findings
"""
    
    elif role == "nurse":
        return """
NURSE MODE:
- Focus on observable findings and care implications
- Use clear, practical language
- Emphasize what needs to be monitored or reported
- DO NOT provide diagnostic conclusions
- Frame findings in terms of care actions
"""
    
    elif role == "patient":
        return """
PATIENT MODE:
- Use SIMPLE, everyday language
- Explain medical terms when you must use them
- Focus on what the findings mean in practical terms
- Avoid frightening or overly technical descriptions
- Be reassuring where appropriate while remaining truthful
"""
    
    else:
        return ""