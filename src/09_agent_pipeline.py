# src/09_agent_pipeline.py
import json
import time
import os

from orchestrator import orchestrate
from reasoner_deepseek import deepseek_reason
from verifier import run_verification

def run_pipeline(query, role="doctor", top_k_text=5, top_k_img=3, log_file="logs/last_run.json"):
    # Orchestrate retrieval + RBAC
    lead = orchestrate(query, role=role, top_k_text=top_k_text, top_k_img=top_k_img)
    retrieved = lead["retrieved"]
    print(f"Retrieved {len(retrieved)} items after RBAC.\n")

    # Reasoner
    start = time.time()
    reasoner_out = deepseek_reason(query, retrieved)
    took = time.time() - start
    print("\n=== REASONER OUTPUT ===\n")
    print(reasoner_out)

    # Verification
    verify_res = run_verification(reasoner_out, retrieved)
    print("\n=== VERIFICATION ===\n")
    for r in verify_res:
        print(r)

    # save log
    os.makedirs("logs", exist_ok=True)
    result = {
        "query": query,
        "role": role,
        "retrieved": retrieved,
        "reasoner_out": reasoner_out,
        "verification": verify_res,
        "time_seconds": took
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, default=str, indent=2)
    print(f"\nSaved pipeline log to {log_file}")
    return result

if __name__ == "__main__":
    q = input("Enter query: ").strip()
    r = input("Enter role (doctor/nurse/patient/admin) [doctor]: ").strip() or "doctor"
    run_pipeline(q, role=r)
