# agents/clinical_reasoning_agent.py

# agents/clinical_reasoning_agent.py

import os
import requests
import psutil
import subprocess
import time

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

    ensure_ollama_running()
    #logger.info("Extracting image insights using PaliGemma")
    image_insights = image_insight_agent_ollama(evidence)

    combined_evidence = []
    for i, e in enumerate(evidence, start=1):
        combined_evidence.append(
            f"[R{i}] ({e['modality']}) {e['report_text']}"
        )

    combined_evidence.extend(image_insights)

    prompt = f"""
You are a clinical decision-support AI.

Rules:
- Use ONLY the provided evidence
- Do NOT hallucinate diagnoses
- If evidence is insufficient, say so clearly
- Cite evidence as [R1], [R2], [R1-IMAGE], etc.

Retrieved Clinical Evidence:
{chr(10).join(combined_evidence)}

Clinical Question:
{query}

Provide:
1. Step-by-step reasoning
2. Final concise answer
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
    return response.json()["response"]

