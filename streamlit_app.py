# streamlit_app.py

import streamlit as st
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from PIL import Image
import os
import pandas as pd
import json
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MM-RAG Clinical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quality-badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .quality-badge-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stage-box {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header"> Multimodal Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered clinical reasoning with Quality Gates & Local Feedback Loops</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR — thresholds & retries only
# ============================================================

with st.sidebar:
    st.header(" Configuration")

    st.subheader("Quality Thresholds")
    evidence_threshold = st.slider("Evidence Quality", 0.0, 1.0, 0.6, 0.05)
    response_threshold = st.slider("Response Quality", 0.0, 1.0, 0.7, 0.05)

    st.subheader("Retry Limits")
    max_retrieval_retries = st.number_input("Max Retrieval Retries", 1, 5, 2)
    max_reasoning_retries = st.number_input("Max Reasoning Retries", 1, 5, 2)

    st.markdown("**ℹ About**")
    st.markdown("""
    This system uses:
    - Multi-agent architecture
    - Local feedback loops
    - Quality gates between stages
    - Evidence pre-filtering
    - Progressive refinement
    - Pathology detection (DenseNet-121)
    """)



# ============================================================
# INPUT
# ============================================================

st.markdown("---")
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
        height=100
    )

run_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)

# ============================================================
# RUN PIPELINE
# ============================================================

if run_button and query.strip():

    with st.spinner(" Running multi-agent pipeline with quality gates..."):

        graph = build_mmrag_graph()

        initial_state = {
            "patient_id": int(patient_id),
            "query": query,


            # Routing
            "modalities": ["XRAY"],
            "routing_verification": {},
            "routing_gate_result": {},

            # Retrieval
            "xray_results": [],
            "ct_results": [],
            "mri_results": [],

            # Evidence
            "evidence": [],
            "filtered_evidence": [],
            "evidence_filter_result": {},
            "evidence_gate_result": {},
            "retrieval_attempts": 0,

            # Reasoning
            "final_answer": "",
            "metrics": {},
            "response_gate_result": {},
            "refinement_result": {},
            "reasoning_attempts": 0,
            "refinement_count": 0,

            # Global
            "total_iterations": 0,
            "quality_scores": {}
        }

        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit":50}
        )

    st.success(" Pipeline completed!")

    # ============================================================
    # PIPELINE SUMMARY
    # ============================================================

    st.markdown("---")
    st.header(" Pipeline Execution Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_iterations = final_state.get('total_iterations', 0)
        st.metric("Total Iterations", total_iterations)

    with col2:
        retrieval_attempts = final_state.get('retrieval_attempts', 0)
        st.metric("Retrieval Attempts", retrieval_attempts)

    with col3:
        reasoning_attempts = final_state.get('reasoning_attempts', 0)
        st.metric("Reasoning Attempts", reasoning_attempts)

    # ============================================================
    # QUALITY GATES
    # ============================================================

    st.markdown("---")
    st.header("✅ Quality Gate Results")

    quality_scores = final_state.get('quality_scores', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        evidence_score = quality_scores.get('evidence', 0)
        st.metric(
            "Evidence Quality",
            f"{evidence_score:.2f}",
            delta=f"{evidence_score - evidence_threshold:.2f}" if evidence_score else None
        )
        if evidence_score >= evidence_threshold:
            st.markdown('<div class="quality-badge-pass"> PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail"> ATTENTION</div>', unsafe_allow_html=True)

    with col2:
        response_score = quality_scores.get('response', 0)
        st.metric(
            "Response Quality",
            f"{response_score:.2f}",
            delta=f"{response_score - response_threshold:.2f}" if response_score else None
        )
        if response_score >= response_threshold:
            st.markdown('<div class="quality-badge-pass"> PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail"> ATTENTION</div>', unsafe_allow_html=True)

    with col3:
        avg_quality = (
            (quality_scores.get("evidence", 0) + quality_scores.get("response", 0)) / 2
        )
        st.metric("Overall Quality", f"{avg_quality:.2f}")
        if avg_quality >= 0.7:
            st.markdown('<div class="quality-badge-pass"> EXCELLENT</div>', unsafe_allow_html=True)
        elif avg_quality >= 0.5:
            st.markdown('<div class="quality-badge-pass"> GOOD</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail"> NEEDS REVIEW</div>', unsafe_allow_html=True)

    # ============================================================
    # STAGE BREAKDOWN
    # ============================================================

    st.markdown("---")
    st.header(" Stage-by-Stage Breakdown")

    # Stage 1: Evidence Retrieval
    with st.expander("**Stage 1: Evidence Retrieval**", expanded=False):
        evidence_gate = final_state.get('evidence_gate_result', {})
        filter_result = final_state.get('evidence_filter_result', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            original_count = len(final_state.get('evidence', []))
            st.metric("Original Evidence", original_count)

        with col2:
            filtered_count = len(final_state.get('filtered_evidence', []))
            st.metric("Filtered Evidence", filtered_count)

        with col3:
            removed_count = filter_result.get('removed_count', 0)
            st.metric("Removed", removed_count, delta=f"-{removed_count}")

        st.markdown("**Filter Quality:**")
        st.write(f"Quality Score: {filter_result.get('quality_score', 0):.2f}")
        st.write(f"Gate Decision: {evidence_gate.get('decision', 'N/A')}")

        if filter_result.get('feedback'):
            st.info(filter_result['feedback'])

    # Stage 3: Clinical Reasoning
    with st.expander("**Stage 3: Clinical Reasoning** ", expanded=False):
        response_gate = final_state.get('response_gate_result', {})
        refinement = final_state.get('refinement_result', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Response Quality:**")
            st.write(f"Score: {response_gate.get('score', 0):.2f}")
            st.write(f"Decision: {response_gate.get('decision', 'N/A')}")

        with col2:
            if refinement:
                st.markdown("**Refinements Applied:**")
                refinements = refinement.get('refinements_applied', [])
                if refinements:
                    for ref in refinements:
                        st.markdown(f"- {ref.replace('_', ' ').title()}")
                else:
                    st.write("No refinements needed")

    # ============================================================
    # RETRIEVED EVIDENCE
    # ============================================================

    st.markdown("---")
    st.header(" Retrieved Evidence")

    filtered_evidence = final_state.get("filtered_evidence", [])

    if not filtered_evidence:
        st.warning("⚠️ No relevant evidence found")
    else:
        st.success(f"✅ Found {len(filtered_evidence)} relevant evidence items")

        modality_counts = {}
        for e in filtered_evidence:
            mod = e.get('modality', 'Unknown')
            modality_counts[mod] = modality_counts.get(mod, 0) + 1

        st.markdown("**Evidence by Modality:**")
        cols = st.columns(len(modality_counts))
        for col, (mod, count) in zip(cols, modality_counts.items()):
            col.metric(mod, count)

        st.markdown("---")

        for idx, e in enumerate(filtered_evidence, start=1):
            with st.expander(f"**Evidence {idx}** — {e.get('modality', 'N/A')} (Relevance: {e.get('relevance_score', 0):.2f})"):

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Report Text:**")
                    st.write(e.get('report_text', 'N/A'))
                    st.markdown(f"**Organ:** {e.get('organ', 'N/A')}")
                    st.markdown(f"**Modality:** {e.get('modality', 'N/A')}")

                with col2:
                    # RBAC: nurses have has_image=False from rbac_filter
                    if e.get('has_image') and e.get('image_path') and os.path.exists(e['image_path']):
                        try:
                            img = Image.open(e['image_path'])
                            st.image(img, caption=f"Image {idx}", use_container_width=True)
                        except Exception as ex:
                            st.warning(f"Unable to display image: {ex}")
                    else:
                        st.info("No image available")


    # ============================================================
    # PATHOLOGY DETECTION RESULTS (NEW SECTION)
    # ============================================================

    st.markdown("---")
    st.header(" Pathology Detection Results")

    # Check if any evidence has pathology scores
    has_pathology_data = any(
        "pathology_scores" in e and e["pathology_scores"]
        for e in filtered_evidence
    )

    max_scores = {}
    sorted_pathologies = []

    if has_pathology_data:
        
        # Aggregate all pathology scores across evidence
        aggregated_scores = {}
        for e in filtered_evidence:
            if "pathology_scores" in e:
                for pathology, score in e["pathology_scores"].items():
                    if pathology not in aggregated_scores:
                        aggregated_scores[pathology] = []
                    aggregated_scores[pathology].append(score)
        
        # Calculate max score per pathology
        max_scores = {
            pathology: max(scores)
            for pathology, scores in aggregated_scores.items()
        }
        
        # Sort by score
        sorted_pathologies = sorted(
            max_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:8]  # Top 8
        
        # Create bar chart
        if sorted_pathologies:
            import plotly.graph_objects as go
            
            pathology_names = [p[0] for p in sorted_pathologies]
            scores = [p[1] * 100 for p in sorted_pathologies]  # Convert to percentage
            
            # Color based on confidence
            colors = [
                'rgb(220, 53, 69)' if s >= 70 else    # Red for high
                'rgb(255, 193, 7)' if s >= 50 else    # Yellow for moderate
                'rgb(40, 167, 69)'                     # Green for low
                for s in scores
            ]
            
            fig = go.Figure(go.Bar(
                x=scores,
                y=pathology_names,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{s:.1f}%' for s in scores],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Detected Pathologies (Maximum Confidence Across All Images)",
                xaxis_title="Confidence (%)",
                yaxis_title="Pathology",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add legend for confidence levels
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🔴 **High Confidence** (≥70%)")
            with col2:
                st.markdown("🟡 **Moderate Confidence** (40-70%)")
            with col3:
                st.markdown("🟢 **Low Confidence** (<40%)")
        
        # Show detailed findings per evidence item
        st.markdown("### Detailed Findings by Image")
        
        for idx, e in enumerate(filtered_evidence, start=1):
            if "pathology_findings" in e and e["pathology_findings"]:
                with st.expander(f"Evidence {idx} - Pathology Analysis"):
                    st.markdown(e["pathology_findings"])
                    
                    # Show scores if available (for doctors)
                    # Show scores if available (for doctors)
                    if "pathology_scores" in e:
                        st.markdown("**Detailed Scores:**")
                        
                        # ✅ Sort by probability (highest first) and show top 10
                        sorted_scores = sorted(
                            e["pathology_scores"].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # ✅ Filter: show anything above 1% (instead of 10%)
                        filtered_scores = [
                            (pathology, score) 
                            for pathology, score in sorted_scores 
                            if score > 0.01  # 1% threshold
                        ][:10]  # Top 10 max
                        
                        if filtered_scores:
                            scores_df = pd.DataFrame([
                                {
                                    "Pathology": pathology, 
                                    "Probability": f"{score*100:.2f}%",
                                    "Confidence": (
                                        "🔴 High" if score >= 0.7 else 
                                        "🟡 Moderate" if score >= 0.5 else 
                                        "🟢 Low"
                                    )
                                }
                                for pathology, score in filtered_scores
                            ])
                            st.dataframe(scores_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("ℹ All pathology scores below 1% threshold")

    else:
        st.info("No pathology detection data available for this query.")

    # ============================================================
    # FINAL CLINICAL RESPONSE
    # ============================================================

    st.markdown("---")
    st.header("Final Clinical Response")

    final_answer = final_state.get("final_answer", "No response generated")

    sections = {
        "diagnosis": "",
        "evidence": "",
        "recommendations": ""
    }

    current_section = None
    lines = final_answer.split('\n')

    for line in lines:
        line_lower = line.lower().strip()

        if 'diagnosis' in line_lower or 'impression' in line_lower:
            current_section = "diagnosis"
            continue
        elif 'supporting evidence' in line_lower or 'evidence' in line_lower:
            current_section = "evidence"
            continue
        elif 'next steps' in line_lower or 'recommendation' in line_lower:
            current_section = "recommendations"
            continue

        if current_section and line.strip():
            sections[current_section] += line + "\n"

    if sections["diagnosis"]:
        st.markdown("**Diagnosis / Impression:**")
        st.info(sections["diagnosis"].strip())

    if sections["evidence"]:
        st.markdown("**Supporting Evidence:**")
        evidence_lines = [l.strip() for l in sections["evidence"].split('\n') if l.strip()]
        for evidence_line in evidence_lines:
            st.markdown(evidence_line)

    if sections["recommendations"]:
        st.markdown("**Next Steps / Recommendations:**")
        rec_lines = [l.strip() for l in sections["recommendations"].split('\n') if l.strip()]
        for rec_line in rec_lines:
            st.markdown(rec_line)

    if not any(sections.values()):
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                {final_answer.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

    refinement = final_state.get('refinement_result', {})
    if refinement and refinement.get('refinements_applied'):
        st.info(f"ℹ This response was refined through {refinement.get('iterations', 0)} stages: {', '.join(refinement.get('refinements_applied', []))}")

    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    st.markdown("---")
    st.header(" Evaluation Metrics")

    metrics = final_state.get("metrics", {})

    if metrics:
        metric_cols = st.columns(len(metrics))
        for col, (k, v) in zip(metric_cols, metrics.items()):
            col.metric(label=k, value=f"{v:.3f}" if isinstance(v, float) else v)
    else:
        st.warning("No evaluation metrics available")

    # ============================================================
    # EXPORT
    # ============================================================

    st.markdown("---")
    st.header(" Export Results")

    col1, col2 = st.columns(2)

    with col1:
        export_text = f"""
CLINICAL DECISION SUPPORT REPORT
{'='*60}

Patient ID: {patient_id}
Query: {query}

PIPELINE SUMMARY
{'-'*60}
Total Iterations: {total_iterations}
Retrieval Attempts: {retrieval_attempts}
Reasoning Attempts: {reasoning_attempts}

QUALITY SCORES
{'-'*60}
Evidence Quality: {evidence_score:.2f}
Response Quality: {response_score:.2f}
Overall Quality: {avg_quality:.2f}

CLINICAL RESPONSE
{'-'*60}
{final_answer}

EVALUATION METRICS
{'-'*60}
{chr(10).join([f'{k}: {v}' for k, v in metrics.items()])}
"""

        st.download_button(
            label=" Download as Text",
            data=export_text,
            file_name=f"clinical_report_patient_{patient_id}.txt",
            mime="text/plain"
        )

    with col2:
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_metrics = convert_to_serializable(metrics)
        serializable_quality_scores = convert_to_serializable(quality_scores)

        # 1. Create the base data dictionary
        export_json_data = {
            "patient_id": patient_id,
            "query": query,
            "pipeline_summary": {
                "total_iterations": int(total_iterations),
                "retrieval_attempts": int(retrieval_attempts),
                "reasoning_attempts": int(reasoning_attempts)
            },
            "quality_scores": serializable_quality_scores,
            "final_answer": final_answer,
            "metrics": serializable_metrics,
            "evidence_count": len(filtered_evidence)
        }

        # 2. Add pathology data if available (FIXED)
        if has_pathology_data:
            pathology_export = {
                "detected_pathologies": max_scores,
                "top_findings": [
                    {"pathology": p[0], "confidence": f"{p[1]*100:.1f}%"}
                    for p in sorted_pathologies[:5]
                ]
            }
            export_json_data["pathology_detection"] = pathology_export

        # 3. Convert to JSON string
        export_json = json.dumps(export_json_data, indent=2)

        st.download_button(
            label=" Download as JSON",
            data=export_json,
            file_name=f"clinical_report_patient_{patient_id}.json",
            mime="application/json"
        )

else:
    # ============================================================
    # PLACEHOLDER
    # ============================================================

    st.info(" Enter **Patient ID** and **Clinical Query**, then click **Run Analysis** to start.")

    st.markdown("---")
    st.markdown("###  Example Queries")

    examples = [
        "Is there any pulmonary abnormality?",
        "Are there signs of pleural effusion?",
        "Is there evidence of cardiomegaly?",
        "What are the findings in the chest X-ray?",
    ]

    for example in examples:
        if st.button(f" {example}", key=example):
            st.session_state['example_query'] = example
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
         Built for Clinical AI Research | Powered by LangGraph & Ollama
    </div>
    """,
    unsafe_allow_html=True
)