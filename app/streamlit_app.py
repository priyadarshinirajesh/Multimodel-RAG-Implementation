# src/streamlit_app.py

import streamlit as st
import json
import sys
import os

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

if run_button and query.strip():

    st.info("Running retrieval...")

    lead = orchestrate(query, role=role, top_k_text=5, top_k_img=5)
    retrieved = lead["retrieved"]

    st.subheader("📌 Retrieved Evidence")

    for i, item in enumerate(retrieved):
        st.markdown(f"### Evidence R{i+1}")

        # ---- SHOW IMAGE IN SMALL SIZE ----
        if os.path.exists(item["filename"]):
            st.image(item["filename"], caption=item["filename"], width=400)
        else:
            st.warning(f"Image file not found: {item['filename']}")

        # ---- SHOW TEXT METADATA ----
        st.json(item)

    # -----------------------------
    # DeepSeek Reasoning
    # -----------------------------
    st.info("Running DeepSeek reasoning...")
    reason_output = deepseek_reason(query, retrieved)

    st.subheader("🧠 Reasoner Output (DeepSeek-R1)")
    st.write(reason_output)

    # -----------------------------
    # Verification
    # -----------------------------
    st.info("Running verification module...")
    verification = run_verification(reason_output, retrieved)

    st.subheader("✔ Claim Verification")
    for v in verification:
        st.json(v)

    # Save results
    result = {
        "query": query,
        "role": role,
        "retrieved": retrieved,
        "reasoner_output": reason_output,
        "verification": verification
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/ui_last_run.json", "w") as f:
        json.dump(result, f, indent=2)

    st.success("Pipeline completed!")
    st.download_button(
        "Download Log File",
        data=json.dumps(result, indent=2),
        file_name="rag_result.json",
        mime="application/json"
    )
