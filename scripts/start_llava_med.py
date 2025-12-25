# scripts/start_llava_med.py

import subprocess
import time
import requests
import sys

CONTROLLER_URL = "http://localhost:10000"
WORKER_URL = "http://localhost:40000"

def is_running():
    try:
        r = requests.get(f"{CONTROLLER_URL}/v1/models", timeout=2)
        return r.status_code == 200
    except:
        return False

def start_llava_med():
    print("[INFO] Starting LLaVA-Med controller...")
    subprocess.Popen(
        ["python", "-m", "llava.serve.controller", "--port", "10000"],
        shell=True
    )

    time.sleep(5)

    print("[INFO] Starting LLaVA-Med worker...")
    subprocess.Popen(
        [
            "python", "-m", "llava.serve.model_worker",
            "--controller", CONTROLLER_URL,
            "--port", "40000",
            "--worker", WORKER_URL,
            "--model-path", "microsoft/llava-med-v1.5-mistral-7b",
            "--multi-modal"
        ],
        shell=True
    )

    # Wait until ready
    for _ in range(30):
        if is_running():
            print("[INFO] LLaVA-Med is ready")
            return
        time.sleep(2)

    print("❌ LLaVA-Med failed to start")
    sys.exit(1)
