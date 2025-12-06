# src/reasoner_deepseek.py
import requests
import json
import time
import subprocess
import psutil
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
BATCH_SCRIPT = os.path.abspath("start_ollama.bat")

def is_ollama_running():
    """Check if Ollama server process is alive."""
    for proc in psutil.process_iter(attrs=['name']):
        if proc.info['name'] and "ollama" in proc.info['name'].lower():
            return True
    return False

def ensure_ollama_running():
    """Ensure Ollama server is running — else auto-start it."""
    if is_ollama_running():
        print("✅ Ollama is already running.")
        return

    print("⚠️ Ollama is NOT running. Starting it now...")
    subprocess.Popen([BATCH_SCRIPT], shell=True)

    # Wait until server responds
    for _ in range(20):
        try:
            r = requests.get(OLLAMA_URL, timeout=2)
            if r.status_code == 200:
                print("✅ Ollama server is now running!")
                return
        except:
            pass
        print("⏳ Waiting for Ollama to start...")
        time.sleep(1)

    raise RuntimeError("❌ Ollama did not start. Please start it manually.")

def deepseek_reason(query, retrieved):
    ensure_ollama_running()

    # Build evidence
    evidence_lines = []
    for i, item in enumerate(retrieved):
        eid = f"R{i+1}"
        evidence_lines.append(
            f"{eid} | modality: {item.get('modality')} | findings: {item.get('findings','')} | impression: {item.get('impression','')}"
        )
    evidence_text = "\n".join(evidence_lines)

    prompt = f"""
You are a medical reasoning agent. Use ONLY the retrieved evidence below.

Retrieved evidence:
{evidence_text}

User query:
{query}

Task:
1. Provide step-by-step reasoning with citations like [R1], [R2].
2. Give a final conclusion labelled "Final Answer:".
"""

    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()["response"]
    return result
