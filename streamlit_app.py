# streamlit_app.py (UPDATED)

import streamlit as st
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from PIL import Image
import os

st.set_page_config(
    page_title="Multimodal Clinical Decision Support System",
    layout="wide"
)

st.title("🧠 Multimodal Clinical Decision Support System")
st.markdown(
    "AI-powered multimodal clinical decision support using XRAY, CT, and MRI evidence with **Verification Agent**."
)
st.markdown("---")

# Input Section
st.subheader("📝 Input Parameters")

col1, col2 = st.columns([1, 3])

with col1:
    patient_id = st.number_input(
        "Patient ID",
        min_value=1,
        step=1,
        value=1
    )

with col2:
    query = st.text_area(
        "Clinical Query",
        placeholder="e.g., Are there signs of pleural effusion in the X-ray?",
        height=80
    )

run_button = st.button("🔬 Run Analysis")

st.markdown("---")

# Run pipeline
if run_button and query.strip():

    with st.spinner("Running multimodal RAG pipeline with verification..."):
        graph = build_mmrag_graph()

        initial_state = {
            "patient_id": int(patient_id),
            "query": query,
            "modalities": [],
            "xray_results": [],
            "ct_results": [],
            "mri_results": [],
            "evidence": [],
            "final_answer": "",
            "metrics": {},
            "verification_result": {},
            "improvement_suggestions": [],
            "requires_rerun": False,
            "rerun_count": 0
        }

        final_state = graph.invoke(initial_state)

    # Verification Results (TOP)
    st.subheader("✅ Verification Results")
    
    verification = final_state.get("verification_result", {})
    
    if verification:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pass_status = verification.get('overall_pass', False)
            st.metric(
                "Overall Status",
                "✅ PASS" if pass_status else "⚠️ ATTENTION",
                delta=None
            )
        
        with col2:
            confidence = verification.get('confidence_score', 0)
            st.metric(
                "Confidence Score",
                f"{confidence:.1%}",
                delta=None
            )
        
        with col3:
            reruns = final_state.get('rerun_count', 0)
            st.metric(
                "Reruns Performed",
                reruns,
                delta=None
            )
        
        # Component scores
        st.markdown("**Component Scores:**")
        
        score_cols = st.columns(4)
        
        with score_cols[0]:
            mod_score = verification.get('modality_routing', {}).get('score', 0)
            st.metric("Modality Routing", f"{mod_score:.2f}")
        
        with score_cols[1]:
            ev_score = verification.get('evidence_quality', {}).get('score', 0)
            st.metric("Evidence Quality", f"{ev_score:.2f}")
        
        with score_cols[2]:
            clin_score = verification.get('clinical_response', {}).get('score', 0)
            st.metric("Clinical Response", f"{clin_score:.2f}")
        
        with score_cols[3]:
            cite_score = verification.get('citation_check', {}).get('score', 0)
            st.metric("Citation Check", f"{cite_score:.2f}")
        
        # Improvement suggestions
        suggestions = final_state.get("improvement_suggestions", [])
        if suggestions:
            st.markdown("**⚠️ Improvement Suggestions Applied:**")
            for i, s in enumerate(suggestions, 1):
                priority_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(s['priority'], "⚪")
                st.markdown(f"{i}. {priority_emoji} **{s['agent']}**: {s['details']}")
        else:
            st.success("✓ No improvements needed - pipeline executed correctly")
    
    st.markdown("---")
    
    # Retrieved Evidence
    st.subheader("🔎 Retrieved Evidence")

    if not final_state["evidence"]:
        st.warning("No evidence retrieved.")
    else:
        for idx, e in enumerate(final_state["evidence"], start=1):
            with st.expander(f"Evidence {idx} — {e['modality']}"):
                st.markdown(f"**Modality:** {e['modality']}")
                st.markdown(f"**Organ:** {e.get('organ', 'N/A')}")
                st.markdown("**Report Text:**")
                st.write(e["report_text"])

                if e.get("image_path") and os.path.exists(e["image_path"]):
                    try:
                        img = Image.open(e["image_path"])
                        st.image(
                            img,
                            caption=os.path.basename(e["image_path"]),
                            use_column_width=True
                        )
                    except Exception as ex:
                        st.warning(f"Unable to display image: {ex}")

    st.markdown("---")

    # Final Clinical Response
    st.subheader("🧠 Final Clinical Response")
    st.markdown(final_state["final_answer"])

    st.markdown("---")

    # Evaluation Metrics
    st.subheader("📊 Evaluation Metrics")

    if final_state["metrics"]:
        metric_cols = st.columns(len(final_state["metrics"]))
        for col, (k, v) in zip(metric_cols, final_state["metrics"].items()):
            col.metric(label=k, value=v)
    else:
        st.warning("No evaluation metrics available.")

else:
    st.info("Please enter **Patient ID** and **Clinical Query**, then click **Run Analysis**.")