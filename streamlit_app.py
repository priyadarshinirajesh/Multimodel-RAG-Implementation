# streamlit_app.py

import streamlit as st
import os
import sys

# Let Python import /src
sys.path.append(os.path.abspath("src"))

# Import LangGraph controller
from agent_graph import AgentGraph

# Old evaluation + verifier
from evaluation import evaluate
from verifier import run_verification


st.title("🩺 Multi‑Agent Medical Diagnosis Assistant (LangGraph‑Style)")

# --------------------------
# USER INPUT
# --------------------------
patient_id = st.text_input("Patient ID:")
query = st.text_input("Query:")
role = st.selectbox("Role:", ["doctor", "nurse", "patient", "admin"])


# --------------------------
# RUN BUTTON
# --------------------------
if st.button("Run Analysis"):

    st.write("### 🚀 Running Multi‑Agent Pipeline…\n")

    graph = AgentGraph()

    # IMPORTANT — AgentGraph returns a dictionary:
    # { "answer": ..., "retrieved": ..., "ground_truth": ... }
    result = graph.run(query=query, patient_id=patient_id, role=role)

    answer = result["answer"]
    retrieved = result["retrieved"]
    ground_truth = result["ground_truth"]

    # --------------------------
    # STREAMLIT DEBUG LOG OUTPUT
    # --------------------------
    print("\n=========== STREAMLIT DEBUG ===========")
    print("Result received from AgentGraph:")
    print(result)
    print("=======================================\n")


    # -----------------------------------------------------
    # IMAGE DISPLAY (same debug logs you previously used)
    # -----------------------------------------------------
    st.subheader("📸 Patient Imaging")

    found = False
    for r in retrieved:
        if "filename" in r:
            print("\n[IMAGE DEBUG] Checking:", r["filename"])

            if os.path.exists(r["filename"]):
                st.image(r["filename"], caption=r.get("modality", "Image"))
                found = True
            else:
                print("❌ File does NOT exist:", r["filename"])

    if not found:
        st.warning("No images found for this patient.")


    # -----------------------------------------------------
    # FINAL DIAGNOSIS
    # -----------------------------------------------------
    st.subheader("🧠 Final Diagnosis (Reasoner Agent)")
    st.markdown(answer)


    # -----------------------------------------------------
    # EVALUATION METRICS
    # -----------------------------------------------------
    metrics = evaluate(answer, retrieved, ground_truth)

    st.subheader("📊 Evaluation Metrics")
    st.write(f"**Semantic Similarity:** {metrics['semantic_similarity']:.4f}")
    st.write(f"**Faithfulness Score:** {metrics['faithfulness']:.4f}")
    st.write(f"**Retrieved Evidence Count:** {metrics['retrieval_count']}")
    st.write(f"**Hallucination Rate:** {metrics['hallucination_rate']:.2f}")
    
    # Same console debugging as your old RAG system
    print("\n===== EVALUATION METRICS =====")
    print(metrics)
    print("==============================")

    # -----------------------------------------------------
    # RUN CONSOLE VERIFIER (your old traditional verifier)
    # -----------------------------------------------------
    run_verification(answer, retrieved)

    st.success("✔ Multi‑Agent Diagnosis Complete!")