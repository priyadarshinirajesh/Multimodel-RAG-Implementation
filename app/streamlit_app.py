# app/streamlit_app.py
import streamlit as st
import json
import os
import sys

sys.path.append(os.path.abspath("src"))

from orchestrator import orchestrate
from reasoner_deepseek import deepseek_reason
from verifier import run_verification
from evaluation import evaluate_all

st.title("🩺 Medical Diagnosis Assistant")

patient_id = st.text_input("Patient ID:")
query = st.text_input("Query:")
role = st.selectbox("Role:", ["doctor", "nurse", "patient", "admin"])

if st.button("Run Analysis"):
    result = orchestrate(query, role, patient_id)
    retrieved = result["retrieved"]

    # ------------------------------
    # IMAGE DISPLAY
    # ------------------------------
    st.subheader("📸 Patient Imaging")

    found = False
    for r in retrieved:
        if "filename" in r:
            print("\n[IMAGE DEBUG] Checking:", r["filename"])
            if os.path.exists(r["filename"]):
                st.image(r["filename"], caption=r.get("modality","Image"))
                found = True
            else:
                print("❌ File does NOT exist:", r["filename"])

    if not found:
        st.warning("No images found.")

    # ------------------------------
    # DEEPSEEK
    # ------------------------------
    answer = deepseek_reason(query, retrieved)
    st.subheader("🧠 Final Diagnosis")
    st.markdown(answer)

    # ------------------------------
    # TERMINAL VERIFICATION + METRICS
    # ------------------------------
    run_verification(answer, retrieved)

    reference_text = " ".join([i.get("findings","") for i in retrieved])
    evaluate_all(reference_text, answer, retrieved)

    st.success("Analysis complete.")
