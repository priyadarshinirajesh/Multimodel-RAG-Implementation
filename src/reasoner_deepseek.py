# src/reasoner_deepseek.py
import os
import requests
import json
import psutil
import subprocess
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
BATCH_SCRIPT = os.path.abspath("start_ollama.bat")

def is_ollama_running():
    for p in psutil.process_iter(attrs=["name"]):
        if "ollama" in (p.info["name"] or "").lower():
            return True
    return False

def ensure_ollama_running():
    if is_ollama_running():
        print("✅ Ollama already running.")
        return
    print("⚠️ Ollama not running → Starting…")
    subprocess.Popen([BATCH_SCRIPT], shell=True)

    for _ in range(20):
        try:
            r = requests.get(OLLAMA_URL, timeout=2)
            if r.status_code == 200:
                print("✅ Ollama started!")
                return
        except:
            pass
        time.sleep(1)
    raise RuntimeError("❌ Ollama failed to start.")

def deepseek_reason(query, retrieved):
    ensure_ollama_running()

    # Build evidence
    print("\n================ DEEPSEEK DEBUG ================")
    print("User Query:", query)

    evidence = []
    for idx, item in enumerate(retrieved):
        print(f"[EVIDENCE {idx+1}] {item}")
        evidence.append(
            f"R{idx+1}: modality={item.get('modality')} findings={item.get('findings','')} impression={item.get('impression','')}"
        )

    evidence_text = "\n".join(evidence)

    prompt = f"""
You are a medical reasoning agent.
Use ONLY the retrieved evidence below.

Retrieved Evidence:
{evidence_text}

User Question:
{query}

Provide step-by-step reasoning using citations [R1], [R2] and a final answer.
"""

    print("\n--- FINAL PROMPT SENT TO DEEPSEEK ---")
    print(prompt)
    print("=====================================================")

    payload = {"model": "deepseek-r1:7b", "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()["response"]
