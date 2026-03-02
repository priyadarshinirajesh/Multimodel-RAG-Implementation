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
    .discordance-note {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .differential-primary {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem 1rem;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.3rem 0;
    }
    .differential-alternative {
        background-color: #e2e3e5;
        border-left: 4px solid #6c757d;
        padding: 0.5rem 1rem;
        border-radius: 0 0.3rem 0.3rem 0;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">Multimodal Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered clinical reasoning with Quality Gates & Local Feedback Loops</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Configuration")

    st.subheader("Quality Thresholds")
    evidence_threshold = st.slider(
        "Evidence Quality Threshold", 0.0, 1.0, 0.4, 0.05,
        help="EvidenceQualityGate pass threshold."
    )
    response_threshold = st.slider(
        "Response Quality Threshold", 0.0, 1.0, 0.7, 0.05,
        help="ResponseQualityGate pass threshold."
    )

    st.subheader("Retry Limits")
    max_retrieval_retries = st.number_input(
        "Max Retrieval Retries", 1, 5, 2,
        help="Maximum xray fetch + aggregation cycles."
    )
    max_refinement_retries = st.number_input(
        "Max Refinement Retries", 1, 5, 2,
        help="Maximum response_refine cycles."
    )

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
st.subheader("Input Parameters")

col1, col2 = st.columns([1, 3])

with col1:
    patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)

with col2:
    query = st.text_area(
        "Clinical Query",
        placeholder="e.g., Are there signs of pleural effusion in the X-ray?",
        height=100
    )

run_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)

# ============================================================
# SECTION PARSER — updated for new four-section format
# ============================================================

def parse_clinical_sections(text: str) -> dict:
    """
    Parse the four-section Clinical Assistant response format.
    Returns a dict with keys: impression, evidence, differential, recommendations.
    All values are lists of line strings (empty list if section not found).
    """
    sections = {
        "impression":      [],
        "evidence":        [],
        "differential":    [],
        "recommendations": [],
    }

    current = None

    for line in text.split("\n"):
        ll = line.lower().strip()

        # ── Detect section headers ────────────────────────────────────────────
        if ll.startswith("clinical impression"):
            current = "impression"
            continue
        elif ll.startswith("evidence synthesis"):
            current = "evidence"
            continue
        elif ll.startswith("differential consideration"):
            current = "differential"
            continue
        elif ll.startswith("actionable next steps"):
            current = "recommendations"
            continue

        # ── Capture content lines ─────────────────────────────────────────────
        if current and line.strip():
            sections[current].append(line.strip())

    return sections


# ============================================================
# RUN PIPELINE
# ============================================================

if run_button and query.strip():

    with st.spinner("Running multi-agent pipeline with quality gates..."):

        graph = build_mmrag_graph()

        initial_state = {
            "patient_id":   int(patient_id),
            "query":        query,
            "user_role":    "doctor",

            "evidence_threshold":     float(evidence_threshold),
            "response_threshold":     float(response_threshold),
            "max_retrieval_retries":  int(max_retrieval_retries),
            "max_refinement_retries": int(max_refinement_retries),

            "modalities":           ["XRAY"],
            "routing_verification": {},
            "routing_gate_result":  {},

            "xray_results": [],
            "ct_results":   [],
            "mri_results":  [],

            "evidence":               [],
            "filtered_evidence":      [],
            "evidence_filter_result": {},
            "evidence_gate_result":   {},
            "retrieval_attempts":     0,

            "final_answer":       "",
            "metrics":            {},
            "response_gate_result": {},
            "refinement_result":    {},
            "reasoning_attempts":   0,
            "refinement_count":     0,
            "forced_complete":      False,

            "total_iterations": 0,
            "quality_scores":   {},
        }

        final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    st.success("Pipeline completed!")

    if final_state.get("forced_complete", False):
        st.warning(
            "⚠️ **Response was force-finalized**: reached maximum refinement retries. "
            "Consider reviewing manually or increasing Max Refinement Retries."
        )

    # ============================================================
    # PIPELINE SUMMARY
    # ============================================================

    st.markdown("---")
    st.header("📊 Pipeline Execution Summary")

    col1, col2, col3, col4 = st.columns(4)

    total_iterations   = final_state.get("total_iterations", 0)
    retrieval_attempts = final_state.get("retrieval_attempts", 0)
    reasoning_attempts = final_state.get("reasoning_attempts", 0)
    refinement_count   = final_state.get("refinement_count", 0)

    col1.metric("Total Iterations",   total_iterations)
    col2.metric("Retrieval Attempts", retrieval_attempts)
    col3.metric("Reasoning Attempts", reasoning_attempts)
    col4.metric("Refinement Count",   refinement_count)

    # ============================================================
    # QUALITY GATES
    # ============================================================

    st.markdown("---")
    st.header("✅ Quality Gate Results")

    quality_scores = final_state.get("quality_scores", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        evidence_score = quality_scores.get("evidence", 0)
        st.metric("Evidence Quality", f"{evidence_score:.2f}",
                  delta=f"{evidence_score - evidence_threshold:.2f}")
        if evidence_score >= evidence_threshold:
            st.markdown('<div class="quality-badge-pass">PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">ATTENTION</div>', unsafe_allow_html=True)

    with col2:
        response_score = quality_scores.get("response", 0)
        st.metric("Response Quality", f"{response_score:.2f}",
                  delta=f"{response_score - response_threshold:.2f}")
        if response_score >= response_threshold:
            st.markdown('<div class="quality-badge-pass">PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">ATTENTION</div>', unsafe_allow_html=True)

    with col3:
        avg_quality = (
            (quality_scores.get("evidence", 0) + quality_scores.get("response", 0)) / 2
        )
        st.metric("Overall Quality", f"{avg_quality:.2f}")
        if avg_quality >= 0.7:
            st.markdown('<div class="quality-badge-pass">EXCELLENT</div>', unsafe_allow_html=True)
        elif avg_quality >= 0.5:
            st.markdown('<div class="quality-badge-pass">GOOD</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">NEEDS REVIEW</div>', unsafe_allow_html=True)

    # ============================================================
    # STAGE BREAKDOWN
    # ============================================================

    st.markdown("---")
    st.header("🔍 Stage-by-Stage Breakdown")

    with st.expander("**Stage 1: Evidence Retrieval**", expanded=False):
        evidence_gate = final_state.get("evidence_gate_result", {})
        filter_result = final_state.get("evidence_filter_result", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Evidence",  len(final_state.get("evidence", [])))
        col2.metric("Filtered Evidence",  len(final_state.get("filtered_evidence", [])))
        removed = filter_result.get("removed_count", 0)
        col3.metric("Removed", removed, delta=f"-{removed}")

        st.write(f"Quality Score: {filter_result.get('quality_score', 0):.2f}")
        st.write(f"Gate Decision: {evidence_gate.get('decision', 'N/A')}")
        if filter_result.get("feedback"):
            st.info(filter_result["feedback"])

        consistency = final_state.get("consistency_result", {})
        if consistency:
            if consistency.get("passed"):
                st.success("✅ Evidence consistency check passed")
            else:
                issues = consistency.get("issues", [])
                st.warning(f"⚠️ Consistency issues: {', '.join(issues)}")

    with st.expander("**Stage 3: Clinical Reasoning**", expanded=False):
        response_gate = final_state.get("response_gate_result", {})
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Score: {response_gate.get('score', 0):.2f}")
            st.write(f"Decision: {response_gate.get('decision', 'N/A')}")

    # ============================================================
    # RETRIEVED EVIDENCE
    # ============================================================

    st.markdown("---")
    st.header("🗂️ Retrieved Evidence")

    filtered_evidence = final_state.get("filtered_evidence", [])

    if not filtered_evidence:
        st.warning("⚠️ No relevant evidence found")
    else:
        st.success(f"✅ Found {len(filtered_evidence)} relevant evidence items")

        modality_counts = {}
        for e in filtered_evidence:
            mod = e.get("modality", "Unknown")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1

        cols = st.columns(len(modality_counts))
        for col, (mod, count) in zip(cols, modality_counts.items()):
            col.metric(mod, count)

        st.markdown("---")

        for idx, e in enumerate(filtered_evidence, start=1):
            with st.expander(
                f"**Evidence {idx}** — {e.get('modality', 'N/A')} "
                f"(Relevance: {e.get('relevance_score', 0):.2f})"
            ):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("**Report Text:**")
                    st.write(e.get("report_text", "N/A"))
                    st.markdown(f"**Organ:** {e.get('organ', 'N/A')}")
                    st.markdown(f"**Modality:** {e.get('modality', 'N/A')}")
                with c2:
                    if e.get("has_image") and e.get("image_path") and os.path.exists(e["image_path"]):
                        try:
                            st.image(
                                Image.open(e["image_path"]),
                                caption=f"Image {idx}",
                                use_container_width=True
                            )
                        except Exception as ex:
                            st.warning(f"Unable to display image: {ex}")
                    else:
                        st.info("No image available")

    # ============================================================
    # PATHOLOGY DETECTION RESULTS
    # ============================================================

    st.markdown("---")
    st.header("🔬 Pathology Detection Results")

    has_pathology_data = any(
        "pathology_scores" in e and e["pathology_scores"]
        for e in filtered_evidence
    )

    max_scores         = {}
    sorted_pathologies = []

    if has_pathology_data:
        aggregated_scores = {}
        for e in filtered_evidence:
            for pathology, score in e.get("pathology_scores", {}).items():
                aggregated_scores.setdefault(pathology, []).append(score)

        max_scores = {p: max(scores) for p, scores in aggregated_scores.items()}
        sorted_pathologies = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)[:8]

        if sorted_pathologies:
            import plotly.graph_objects as go

            names  = [p[0] for p in sorted_pathologies]
            scores = [p[1] * 100 for p in sorted_pathologies]
            colors = [
                "rgb(220,53,69)"  if s >= 70 else
                "rgb(255,193,7)"  if s >= 50 else
                "rgb(40,167,69)"
                for s in scores
            ]

            fig = go.Figure(go.Bar(
                x=scores, y=names, orientation="h",
                marker=dict(color=colors),
                text=[f"{s:.1f}%" for s in scores],
                textposition="auto",
            ))
            fig.update_layout(
                title="Detected Pathologies (Max Confidence Across All Images)",
                xaxis_title="Confidence (%)", yaxis_title="Pathology",
                height=400, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.markdown("🔴 **High Confidence** (≥70%)")
            col2.markdown("🟡 **Moderate Confidence** (40–70%)")
            col3.markdown("🟢 **Low Confidence** (<40%)")

        st.markdown("### Detailed Findings by Image")
        for idx, e in enumerate(filtered_evidence, start=1):
            if e.get("pathology_findings"):
                with st.expander(f"Evidence {idx} - Pathology Analysis"):
                    st.markdown(e["pathology_findings"])
                    if "pathology_scores" in e:
                        sorted_scores = sorted(
                            e["pathology_scores"].items(), key=lambda x: x[1], reverse=True
                        )
                        filtered_scores = [(p, s) for p, s in sorted_scores if s > 0.01][:10]
                        if filtered_scores:
                            st.dataframe(pd.DataFrame([{
                                "Pathology": p,
                                "Probability": f"{s*100:.2f}%",
                                "Confidence": (
                                    "🔴 High" if s >= 0.7 else
                                    "🟡 Moderate" if s >= 0.5 else
                                    "🟢 Low"
                                ),
                            } for p, s in filtered_scores]),
                            hide_index=True, use_container_width=True)
                        else:
                            st.info("ℹ All pathology scores below 1% threshold")
    else:
        st.info("No pathology detection data available for this query.")

    # ============================================================
    # FINAL CLINICAL RESPONSE — four-section format
    # ============================================================

    st.markdown("---")
    st.header("💊 Final Clinical Response")

    final_answer = final_state.get("final_answer", "No response generated")
    sections     = parse_clinical_sections(final_answer)

    # ── 1. Clinical Impression ─────────────────────────────────────────────────
    if sections["impression"]:
        st.markdown("#### 🩺 Clinical Impression")
        for line in sections["impression"]:
            clean = line.lstrip("-• ").strip()
            if clean:
                # Colour-code the confidence badge
                if "high confidence" in clean.lower():
                    st.success(clean)
                elif "moderate confidence" in clean.lower():
                    st.warning(clean)
                elif "low confidence" in clean.lower():
                    st.error(clean)
                else:
                    st.info(clean)

    # ── 2. Evidence Synthesis ──────────────────────────────────────────────────
    if sections["evidence"]:
        st.markdown("#### 🔍 Evidence Synthesis")
        for line in sections["evidence"]:
            clean = line.lstrip("-• ").strip()
            if not clean:
                continue
            if "discordance" in clean.lower():
                st.markdown(
                    f'<div class="discordance-note">⚠️ {clean}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"- {clean}")

    # ── 3. Differential Considerations ────────────────────────────────────────
    if sections["differential"]:
        st.markdown("#### 🔀 Differential Considerations")
        for line in sections["differential"]:
            clean = line.lstrip("-• ").strip()
            if not clean:
                continue
            if clean.lower().startswith("primary"):
                st.markdown(
                    f'<div class="differential-primary">✅ {clean}</div>',
                    unsafe_allow_html=True
                )
            elif clean.lower().startswith("alternative"):
                st.markdown(
                    f'<div class="differential-alternative">🔄 {clean}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"- {clean}")

    # ── 4. Actionable Next Steps — HIDDEN from display as requested ────────────
    # (kept in final_answer for export/metrics, just not shown in UI)

    # ── Fallback: raw response if no sections parsed ───────────────────────────
    if not any([sections["impression"], sections["evidence"], sections["differential"]]):
        st.markdown(
            f'<div style="background-color:#f0f2f6;padding:1.5rem;border-radius:.5rem;'
            f'border-left:4px solid #1f77b4;">'
            f'{final_answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    st.markdown("---")
    st.header("📈 Evaluation Metrics")

    metrics = final_state.get("metrics", {})
    preferred_order = [
        "Precision@K", "Recall@K", "MRR",
        "Groundedness", "ClinicalCorrectness", "Completeness",
        "ActionabilityScore", "ContradictionFlag",
    ]
    hidden_keys = {"GroundednessSimple", "EvaluationNote", "GroundednessSource"}

    visible_metrics = {
        k: metrics[k]
        for k in preferred_order
        if k in metrics and k not in hidden_keys
    }

    if visible_metrics:
        metric_cols = st.columns(len(visible_metrics))
        for col, (k, v) in zip(metric_cols, visible_metrics.items()):
            col.metric(
                label=k,
                value=f"{float(v):.3f}" if isinstance(v, (float, int)) else "N/A"
            )
    else:
        st.warning("No evaluation metrics available")

    # ============================================================
    # EXPORT
    # ============================================================

    st.markdown("---")
    st.header("⬇️ Export Results")

    export_metrics = {
        k: v for k, v in metrics.items()
        if k not in {"GroundednessSimple", "EvaluationNote", "GroundednessSource"}
    }

    col1, col2 = st.columns(2)

    with col1:
        export_text = f"""
CLINICAL DECISION SUPPORT REPORT
{'='*60}

Patient ID: {patient_id}
Query: {query}

PIPELINE SUMMARY
{'-'*60}
Total Iterations:   {total_iterations}
Retrieval Attempts: {retrieval_attempts}
Reasoning Attempts: {reasoning_attempts}
Refinement Count:   {refinement_count}
Force-Finalized:    {final_state.get('forced_complete', False)}

QUALITY SCORES
{'-'*60}
Evidence Quality (threshold={evidence_threshold}): {evidence_score:.2f}
Response Quality (threshold={response_threshold}): {response_score:.2f}
Overall Quality:  {avg_quality:.2f}

CLINICAL RESPONSE
{'-'*60}
{final_answer}

EVALUATION METRICS
{'-'*60}
{chr(10).join([f'{k}: {v}' for k, v in export_metrics.items()])}
"""
        st.download_button(
            label="Download as Text",
            data=export_text,
            file_name=f"clinical_report_patient_{patient_id}.txt",
            mime="text/plain",
        )

    with col2:
        def _to_serializable(obj):
            if isinstance(obj, np.floating):  return float(obj)
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, dict):         return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):         return [_to_serializable(i) for i in obj]
            return obj

        export_json_data = {
            "patient_id": patient_id,
            "query": query,
            "pipeline_summary": {
                "total_iterations":   int(total_iterations),
                "retrieval_attempts": int(retrieval_attempts),
                "reasoning_attempts": int(reasoning_attempts),
                "refinement_count":   int(refinement_count),
                "forced_complete":    bool(final_state.get("forced_complete", False)),
            },
            "quality_scores":  _to_serializable(quality_scores),
            "final_answer":    final_answer,
            "metrics":         _to_serializable(export_metrics),
            "evidence_count":  len(filtered_evidence),
        }

        st.download_button(
            label="Download as JSON",
            data=json.dumps(export_json_data, indent=2),
            file_name=f"clinical_report_patient_{patient_id}.json",
            mime="application/json",
        )

else:
    st.info("Enter **Patient ID** and **Clinical Query**, then click **Run Analysis** to start.")
    st.markdown("---")
    st.markdown("### Example Queries")
    for example in [
        "Is there any pulmonary abnormality?",
        "Are there signs of pleural effusion?",
        "Is there evidence of cardiomegaly?",
        "What are the findings in the chest X-ray?",
    ]:
        if st.button(f"💬 {example}", key=example):
            st.session_state["example_query"] = example
            st.rerun()

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:.9rem;">'
    "Built for Clinical AI Research | Powered by LangGraph & Ollama"
    "</div>",
    unsafe_allow_html=True,
)