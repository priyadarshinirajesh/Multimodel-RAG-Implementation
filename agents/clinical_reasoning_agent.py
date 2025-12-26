# agents/clinical_reasoning_agent.py

import os
import requests
import psutil
import subprocess
import time
from evaluation.diagnosis_evaluator import (
    precision_recall_mrr,
    #role_compliance,
    groundedness,
    clinical_correctness,
    completeness
)

OLLAMA_URL = "http://localhost:11434/api/generate"

BATCH_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "start_ollama.bat")
)


def is_ollama_running():
    for p in psutil.process_iter(attrs=["name"]):
        if "ollama" in (p.info["name"] or "").lower():
            return True
    return False


def ensure_ollama_running():
    if is_ollama_running():
        return

    subprocess.Popen([BATCH_SCRIPT], shell=True)

    for _ in range(20):
        try:
            r = requests.get(OLLAMA_URL, timeout=1)
            if r.status_code == 200:
                return
        except:
            pass
        time.sleep(1)

    raise RuntimeError("❌ Ollama failed to start")

from agents.image_insight_agent_ollama import image_insight_agent_ollama
from utils.logger import get_logger

logger = get_logger("ReasoningAgent")


def clinical_reasoning_agent(query: str, evidence: list):
    logger.info("Starting clinical reasoning")
    logger.info(f"Evidence items received: {len(evidence)}")

    ground_truth_impressions = [
        e["report_text"]
        for e in evidence
        if "impression" in e["report_text"].lower()
    ]

    ensure_ollama_running()
    #logger.info("Extracting image insights using PaliGemma")
    image_insights = image_insight_agent_ollama(evidence,query)

    combined_evidence = []
    for i, e in enumerate(evidence, start=1):
        combined_evidence.append(
            f"[R{i}] ({e['modality']}) {e['report_text']}"
        )

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

Respond in EXACTLY this structure and NOTHING else:

Diagnosis / Impression:
- One short sentence ONLY (max 20 words)
- MUST end with one or more citations like [R1], [R2], [R3]
- IF citation is missing, the answer is INVALID

Supporting Evidence:
- 2–4 bullet points
- Each bullet ≤ 15 words
- Each bullet MUST end with citation

Next Steps / Recommendations:
- 1–2 bullets
- EACH bullet MUST end with a citation [Rx]
- If recommendation is generic, cite the most relevant evidence

IMPORTANT:
- DO NOT mention unrelated organs.
- DO NOT repeat evidence verbatim.
- DO NOT include explanations outside the structure.
"""

    print("---------FINAL PROMPT ---------")
    print(prompt)
    print("-------------------------------")
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    logger.info("Clinical reasoning completed")
    final_answer = response.json()["response"]

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------

    retrieved_texts = [e["report_text"] for e in evidence]

    metrics = {}

    metrics.update(
        precision_recall_mrr(
            retrieved=retrieved_texts,
            ground_truth=ground_truth_impressions,
            k=7
        )
    )

    #metrics["RoleCompliance"] = role_compliance(final_answer)
    metrics["Groundedness"] = groundedness(final_answer)
    metrics["ClinicalCorrectness"] = clinical_correctness(
        final_answer, ground_truth_impressions
    )
    metrics["Completeness"] = completeness(final_answer)

    return {
        "final_answer": final_answer,
        "metrics": metrics
    }


