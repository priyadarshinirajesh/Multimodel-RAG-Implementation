# src/streamlit_app.py
import streamlit as st
import json
import sys
import os
import platform
import subprocess

# Add src folder to Python path
sys.path.append(os.path.abspath("src"))

from orchestrator import orchestrate
from reasoner_deepseek import deepseek_reason
from verifier import run_verification

st.set_page_config(page_title="Medical Agentic RAG", layout="wide")

st.title("🩺 Medical Agentic RAG System (DeepSeek-R1 + Retrieval + Verification)")

query = st.text_input("Enter medical query:")
role = st.selectbox("Select role:", ["doctor", "nurse", "patient", "admin"])
run_button = st.button("Run Agentic RAG Pipeline")

# -------------------------
# Optional: Launch retrieval as a separate terminal process (Windows / POSIX)
# Set to True if you want retrieval to run in a new terminal window
LAUNCH_RETRIEVAL_IN_TERMINAL = False
RETRIEVAL_SCRIPT = os.path.join("src", "05_retrieve.py")  # adjust if different

def launch_retrieval_in_terminal(query_text, role_text):
    """
    Attempts to open a new terminal window and run the retrieval script there.
    Works on Windows (start cmd) or on POSIX systems with xterm/gnome-terminal if available.
    This is optional — set LAUNCH_RETRIEVAL_IN_TERMINAL = True to use.
    """
    system = platform.system().lower()
    cmd = None

    # Pass query & role as env vars or args (here as args)
    args = ["python", RETRIEVAL_SCRIPT, "--query", query_text, "--role", role_text]

    if system == "windows":
        # 'start' opens a new cmd window (shell=True required)
        cmd = f'start cmd /k {" ".join(args)}'
        subprocess.Popen(cmd, shell=True)
    else:
        # try gnome-terminal, xterm, or konsole (best-effort)
        if shutil.which("gnome-terminal"):
            subprocess.Popen(["gnome-terminal", "--"] + args)
        elif shutil.which("xterm"):
            subprocess.Popen(["xterm", "-e"] + args)
        elif shutil.which("konsole"):
            subprocess.Popen(["konsole", "-e"] + args)
        else:
            # fallback: run in background (not a new terminal)
            subprocess.Popen(args)

if run_button and query.strip():

    st.info("Running retrieval (hidden) and generating reasoner answer...")

    # Option A (default): run retrieval inside the Streamlit process (silent).
    #   We call orchestrate(...) but we do NOT display the retrieved items.
    #   This ensures retrieval runs and the reasoner gets the evidence.
    lead = orchestrate(query, role=role, top_k_text=5, top_k_img=5)
    retrieved = lead["retrieved"]

    # Option B: If you prefer the retrieval to run in a separate terminal window,
    # set LAUNCH_RETRIEVAL_IN_TERMINAL = True above. That will launch src/05_retrieve.py
    # in a new terminal instead of calling orchestrate() here.
    # (If you enable that, comment out the orchestrate() call above.)
    if LAUNCH_RETRIEVAL_IN_TERMINAL:
        try:
            launch_retrieval_in_terminal(query, role)
            st.info("Started retrieval in a separate terminal.")
            # you might want to wait or poll for its output; here we proceed to run reasoner using orchestrate anyway
        except Exception as e:
            st.warning(f"Could not launch separate terminal for retrieval: {e}")

    # -------------------------
    # Now run the reasoner and verification (these *are shown* in UI)
    # -------------------------
    with st.spinner("Running DeepSeek reasoner..."):
        try:
            reason_output = deepseek_reason(query, retrieved)
        except Exception as e:
            st.error(f"Reasoner error: {e}")
            reason_output = None

    if reason_output:
        st.subheader("🧠 Reasoner Output (DeepSeek-R1)")
        # Display the text with nicer formatting
        st.markdown(reason_output)

        st.info("Running verification...")
        try:
            verification = run_verification(reason_output, retrieved)
        except Exception as e:
            st.error(f"Verification error: {e}")
            verification = []

        st.subheader("✔ Claim Verification (summary)")
        # show verification as a compact table-like layout
        for v in verification:
            verified_str = "✅" if v.get("verified") else "❌"
            score = v.get("score", 0)
            claim = v.get("claim", "")
            evidence_idx = v.get("evidence_idx", None)
            st.write(f"{verified_str} (score={score:.2f}) — Evidence R{evidence_idx+1 if evidence_idx is not None else 'N/A'} — {claim}")

        # Save last run to logs
        result = {
            "query": query,
            "role": role,
            "reasoner_output": reason_output,
            "verification": verification
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/ui_last_run.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        st.success("Pipeline completed — reasoner output shown above.")
        st.download_button("Download Reasoner & Verification JSON", data=json.dumps(result, indent=2),
                           file_name="rag_reasoner_result.json", mime="application/json")
    else:
        st.error("No reasoner output produced. Check the server/agent logs in the terminal.")
