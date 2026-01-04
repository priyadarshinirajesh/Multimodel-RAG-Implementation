# streamlit_app.py (COMPLETELY UPDATED)

import streamlit as st
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from PIL import Image
import os

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

st.markdown('<div class="main-header">🧠 Multimodal Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered clinical reasoning with Quality Gates & Local Feedback Loops</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Quality Thresholds")
    routing_threshold = st.slider("Routing Quality", 0.0, 1.0, 0.8, 0.05)
    evidence_threshold = st.slider("Evidence Quality", 0.0, 1.0, 0.6, 0.05)
    response_threshold = st.slider("Response Quality", 0.0, 1.0, 0.7, 0.05)
    
    st.subheader("Retry Limits")
    max_routing_retries = st.number_input("Max Routing Retries", 1, 5, 2)
    max_retrieval_retries = st.number_input("Max Retrieval Retries", 1, 5, 2)
    max_reasoning_retries = st.number_input("Max Reasoning Retries", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.markdown("""
    This system uses:
    - Local feedback loops
    - Quality gates between stages
    - Evidence pre-filtering
    - Progressive refinement
    """)

# ============================================================
# MAIN CONTENT - INPUT
# ============================================================

st.markdown("---")
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
        height=100
    )

run_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)

# ============================================================
# RUN PIPELINE
# ============================================================

if run_button and query.strip():
    
    with st.spinner("🔄 Running multi-agent pipeline with quality gates..."):
        
        graph = build_mmrag_graph()
        
        initial_state = {
        "patient_id": int(patient_id),
        "query": query,
        
        # Routing
        "modalities": [],
        "routing_attempts": 0,
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
        "refinement_count": 0,  # ADD THIS LINE
        
        # Global
        "total_iterations": 0,
        "quality_scores": {}
    }
        
        final_state = graph.invoke(initial_state)
    
    st.success("✅ Pipeline completed!")
    
    # ============================================================
    # RESULTS - PIPELINE SUMMARY
    # ============================================================
    
    st.markdown("---")
    st.header("📊 Pipeline Execution Summary")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_iterations = final_state.get('total_iterations', 0)
        st.metric("Total Iterations", total_iterations)
    
    with col2:
        routing_attempts = final_state.get('routing_attempts', 0)
        st.metric("Routing Attempts", routing_attempts)
    
    with col3:
        retrieval_attempts = final_state.get('retrieval_attempts', 0)
        st.metric("Retrieval Attempts", retrieval_attempts)
    
    with col4:
        reasoning_attempts = final_state.get('reasoning_attempts', 0)
        st.metric("Reasoning Attempts", reasoning_attempts)
    
    # ============================================================
    # QUALITY GATES RESULTS
    # ============================================================
    
    st.markdown("---")
    st.header("✅ Quality Gate Results")
    
    quality_scores = final_state.get('quality_scores', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        routing_score = quality_scores.get('routing', 0)
        st.metric(
            "Routing Quality",
            f"{routing_score:.2f}",
            delta=f"{routing_score - routing_threshold:.2f}" if routing_score else None
        )
        if routing_score >= routing_threshold:
            st.markdown('<div class="quality-badge-pass">✅ PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">⚠️ ATTENTION</div>', unsafe_allow_html=True)
    
    with col2:
        evidence_score = quality_scores.get('evidence', 0)
        st.metric(
            "Evidence Quality",
            f"{evidence_score:.2f}",
            delta=f"{evidence_score - evidence_threshold:.2f}" if evidence_score else None
        )
        if evidence_score >= evidence_threshold:
            st.markdown('<div class="quality-badge-pass">✅ PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">⚠️ ATTENTION</div>', unsafe_allow_html=True)
    
    with col3:
        response_score = quality_scores.get('response', 0)
        st.metric(
            "Response Quality",
            f"{response_score:.2f}",
            delta=f"{response_score - response_threshold:.2f}" if response_score else None
        )
        if response_score >= response_threshold:
            st.markdown('<div class="quality-badge-pass">✅ PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">⚠️ ATTENTION</div>', unsafe_allow_html=True)
    
    with col4:
        avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        st.metric("Overall Quality", f"{avg_quality:.2f}")
        if avg_quality >= 0.7:
            st.markdown('<div class="quality-badge-pass">✅ EXCELLENT</div>', unsafe_allow_html=True)
        elif avg_quality >= 0.5:
            st.markdown('<div class="quality-badge-pass">✅ GOOD</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quality-badge-fail">⚠️ NEEDS REVIEW</div>', unsafe_allow_html=True)
    
    # ============================================================
    # STAGE-BY-STAGE BREAKDOWN
    # ============================================================
    
    st.markdown("---")
    st.header("🔄 Stage-by-Stage Breakdown")
    
    # Stage 1: Routing
    with st.expander("**Stage 1: Routing** 🧭", expanded=False):
        routing_gate = final_state.get('routing_gate_result', {})
        routing_verification = final_state.get('routing_verification', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Selected Modalities:**")
            selected_mods = final_state.get('modalities', [])
            for mod in selected_mods:
                st.markdown(f"- {mod}")
        
        with col2:
            st.markdown("**Verification Result:**")
            st.write(f"Confidence: {routing_verification.get('confidence', 0):.2f}")
            st.write(f"Decision: {routing_gate.get('decision', 'N/A')}")
            if routing_gate.get('feedback'):
                st.info(routing_gate['feedback'])
    
    # Stage 2: Evidence Retrieval
    with st.expander("**Stage 2: Evidence Retrieval** 🔍", expanded=False):
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
    with st.expander("**Stage 3: Clinical Reasoning** 🧠", expanded=False):
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
    st.header("🔎 Retrieved Evidence")
    
    filtered_evidence = final_state.get("filtered_evidence", [])
    
    if not filtered_evidence:
        st.warning("⚠️ No relevant evidence found")
    else:
        st.success(f"✅ Found {len(filtered_evidence)} relevant evidence items")
        
        # Evidence statistics
        modality_counts = {}
        for e in filtered_evidence:
            mod = e.get('modality', 'Unknown')
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        st.markdown("**Evidence by Modality:**")
        cols = st.columns(len(modality_counts))
        for col, (mod, count) in zip(cols, modality_counts.items()):
            col.metric(mod, count)
        
        st.markdown("---")
        
        # Display each evidence item
        for idx, e in enumerate(filtered_evidence, start=1):
            with st.expander(f"**Evidence {idx}** — {e.get('modality', 'N/A')} (Relevance: {e.get('relevance_score', 0):.2f})"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Report Text:**")
                    st.write(e.get('report_text', 'N/A'))
                    
                    st.markdown(f"**Organ:** {e.get('organ', 'N/A')}")
                    st.markdown(f"**Modality:** {e.get('modality', 'N/A')}")
                
                with col2:
                    if e.get('image_path') and os.path.exists(e['image_path']):
                        try:
                            img = Image.open(e['image_path'])
                            st.image(
                                img,
                                caption=f"Image {idx}",
                                use_container_width=True
                            )
                        except Exception as ex:
                            st.warning(f"Unable to display image: {ex}")
                    else:
                        st.info("No image available")
    
    # ============================================================
    # FINAL CLINICAL RESPONSE
    # ============================================================

    st.markdown("---")
    st.header("🧠 Final Clinical Response")

    final_answer = final_state.get("final_answer", "No response generated")

    # Parse the response into sections
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

    # Display in organized format
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

    # If parsing failed, show raw response
    if not any(sections.values()):
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                {final_answer.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Show refinement info if applicable
    refinement = final_state.get('refinement_result', {})
    if refinement and refinement.get('refinements_applied'):
        st.info(f"ℹ️ This response was refined through {refinement.get('iterations', 0)} stages: {', '.join(refinement.get('refinements_applied', []))}")
        
    # ============================================================
    # EVALUATION METRICS
    # ============================================================
    
    st.markdown("---")
    st.header("📈 Evaluation Metrics")
    
    metrics = final_state.get("metrics", {})
    
    if metrics:
        metric_cols = st.columns(len(metrics))
        
        for col, (k, v) in zip(metric_cols, metrics.items()):
            col.metric(label=k, value=f"{v:.3f}" if isinstance(v, float) else v)
    else:
        st.warning("No evaluation metrics available")
    
    # ============================================================
    # DOWNLOAD OPTIONS
    # ============================================================
    
    st.markdown("---")
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as text
        export_text = f"""
CLINICAL DECISION SUPPORT REPORT
{'='*60}

Patient ID: {patient_id}
Query: {query}

PIPELINE SUMMARY
{'-'*60}
Total Iterations: {total_iterations}
Routing Attempts: {routing_attempts}
Retrieval Attempts: {retrieval_attempts}
Reasoning Attempts: {reasoning_attempts}

QUALITY SCORES
{'-'*60}
Routing Quality: {routing_score:.2f}
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
            label="📄 Download as Text",
            data=export_text,
            file_name=f"clinical_report_patient_{patient_id}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export as JSON
        import json
        import numpy as np
        
        # Helper function to convert numpy types to Python types
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
        
        # Convert metrics to serializable format
        serializable_metrics = convert_to_serializable(metrics)
        serializable_quality_scores = convert_to_serializable(quality_scores)
        
        export_json = json.dumps({
            "patient_id": patient_id,
            "query": query,
            "pipeline_summary": {
                "total_iterations": int(total_iterations),
                "routing_attempts": int(routing_attempts),
                "retrieval_attempts": int(retrieval_attempts),
                "reasoning_attempts": int(reasoning_attempts)
            },
            "quality_scores": serializable_quality_scores,
            "final_answer": final_answer,
            "metrics": serializable_metrics,
            "evidence_count": len(filtered_evidence)
        }, indent=2)
        
        st.download_button(
            label="📊 Download as JSON",
            data=export_json,
            file_name=f"clinical_report_patient_{patient_id}.json",
            mime="application/json"
        )

else:
    # ============================================================
    # PLACEHOLDER - NO QUERY YET
    # ============================================================
    
    st.info("👆 Enter **Patient ID** and **Clinical Query**, then click **Run Analysis** to start.")
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    examples = [
        "Is there any pulmonary abnormality?",
        "Are there signs of pleural effusion?",
        "Is there evidence of cardiomegaly?",
        "What are the findings in the chest X-ray?",
        "Is there pancreatic duct dilation?",
        "Are there any focal lesions in the prostate?"
    ]
    
    for example in examples:
        if st.button(f"📝 {example}", key=example):
            st.session_state['example_query'] = example
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🏥 Built for Clinical AI Research | Powered by LangGraph & Ollama
    </div>
    """,
    unsafe_allow_html=True
)