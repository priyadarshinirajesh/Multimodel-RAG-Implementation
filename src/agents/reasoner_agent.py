# src/agents/reasoner_agent.py

import os
import requests
import psutil
import subprocess
import time

# Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Path to start_ollama.bat in project root
BATCH_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "start_ollama.bat")
)


def is_ollama_running():
    for p in psutil.process_iter(attrs=["name"]):
        if "ollama" in (p.info["name"] or "").lower():
            return True
    return False


def ensure_ollama_running():
    if is_ollama_running():
        print("✅ Ollama already running.")
        return

    print("⚠️ Ollama not running → starting…")

    subprocess.Popen([BATCH_SCRIPT], shell=True)

    for _ in range(20):
        try:
            r = requests.get(OLLAMA_URL, timeout=1)
            if r.status_code == 200:
                print("✅ Ollama started!")
                return
        except:
            pass
        time.sleep(1)

    raise RuntimeError("❌ Ollama failed to start.")


class ReasonerAgent:
    def run(self, query, evidence, correction=None):
        print("\n[ReasonerAgent] DeepSeek‑7B reasoning started…")

        ensure_ollama_running()

        # BUILD EVIDENCE BLOCK
        evidence_list = []
        print("\n================ DEEPSEEK DEBUG ================")
        print("User Query:", query)

        for i, e in enumerate(evidence, start=1):
            print(f"[EVIDENCE {i}] {e}")
            evidence_list.append(
                f"R{i}: modality={e['modality']} "
                f"findings={e['findings']} "
                f"impression={e['impression']}"
            )

        evidence_text = "\n".join(evidence_list)

        # BUILD PROMPT
        prompt = f"""
You are a medical reasoning agent.
Use ONLY the retrieved evidence below.

Retrieved Evidence:
{evidence_text}

User Question:
{query}
"""

        if correction:
            prompt += f"\nVerifier correction: {correction}\nRevise your answer.\n"

        prompt += """
Provide step-by-step reasoning with citations [R1], [R2], etc., 
and then give a final answer.
"""

        print("\n--- FINAL PROMPT SENT TO DEEPSEEK ---")
        print(prompt)
        print("=====================================================")

        payload = {
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        return response.json()["response"]
