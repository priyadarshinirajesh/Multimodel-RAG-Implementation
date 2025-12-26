import streamlit as st
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from PIL import Image
import os

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(
    page_title="Multimodal Clinical Decision Support System",
    layout="wide"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title(" Multimodal Clinical Decision Support System")
st.markdown(
    "AI-powered multimodal clinical decision support using XRAY, CT, and MRI evidence."
)
st.markdown("---")

# ----------------------------------------------------
# Input Section (MAIN PAGE)
# ----------------------------------------------------
st.subheader(" Input Parameters")

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

run_button = st.button(" Run Analysis")

st.markdown("---")

# ----------------------------------------------------
# Run pipeline
# ----------------------------------------------------
if run_button and query.strip():

    with st.spinner("Running multimodal RAG pipeline..."):
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
            "metrics": {}
        }

        # DEBUG LOGS WILL STILL APPEAR IN TERMINAL
        final_state = graph.invoke(initial_state)

    # ------------------------------------------------
    # Retrieved Evidence
    # ------------------------------------------------
    st.subheader(" Retrieved Evidence")

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

    # ------------------------------------------------
    # Final Clinical Response
    # ------------------------------------------------
    st.subheader(" Final Clinical Response")
    st.markdown(final_state["final_answer"])

    st.markdown("---")

    # ------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------
    st.subheader(" Evaluation Metrics")

    if final_state["metrics"]:
        metric_cols = st.columns(len(final_state["metrics"]))
        for col, (k, v) in zip(metric_cols, final_state["metrics"].items()):
            col.metric(label=k, value=v)
    else:
        st.warning("No evaluation metrics available.")

else:
    st.info("Please enter **Patient ID** and **Clinical Query**, then click **Run Analysis**.")
