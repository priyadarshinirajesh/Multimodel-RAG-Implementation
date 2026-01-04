# app.py (UPDATED FOR IMPROVED SYSTEM)

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph


def main():
    print("\n" + "="*70)
    print("🧠 MULTIMODAL CLINICAL DECISION SUPPORT SYSTEM")
    print("   with Quality Gates & Local Feedback Loops")
    print("="*70 + "\n")

    patient_id = int(input("Enter Patient ID: "))
    query = input("Enter Clinical Query: ")

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

    print("\n🔄 Running multi-agent pipeline with quality gates...\n")
    final_state = graph.invoke(initial_state)

    # ===== DISPLAY RESULTS =====
    
    print("\n" + "="*70)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    print(f"\n🔄 Total Iterations: {final_state.get('total_iterations', 0)}")
    print(f"   • Routing attempts: {final_state.get('routing_attempts', 0)}")
    print(f"   • Retrieval attempts: {final_state.get('retrieval_attempts', 0)}")
    print(f"   • Reasoning attempts: {final_state.get('reasoning_attempts', 0)}")
    
    print("\n✅ Quality Gate Results:")
    quality_scores = final_state.get('quality_scores', {})
    print(f"   • Routing Quality: {quality_scores.get('routing', 0):.2f}")
    print(f"   • Evidence Quality: {quality_scores.get('evidence', 0):.2f}")
    print(f"   • Response Quality: {quality_scores.get('response', 0):.2f}")
    
    avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
    print(f"   • Overall Quality: {avg_quality:.2f}")
    
    print("\n" + "="*70)
    print("🔎 RETRIEVED EVIDENCE")
    print("="*70)
    
    filtered_evidence = final_state.get("filtered_evidence", [])
    if not filtered_evidence:
        print("⚠️  No relevant evidence found")
    else:
        print(f"\nTotal Evidence Items: {len(filtered_evidence)}")
        for i, e in enumerate(filtered_evidence, 1):
            relevance = e.get("relevance_score", 0)
            print(f"\n[Evidence {i}] ({e['modality']}) - Relevance: {relevance:.2f}")
            print(f"  {e['report_text'][:150]}...")
            if e.get("has_image"):
                print(f"  📷 Image: {e.get('image_path', 'N/A')}")
    
    # Evidence filter stats
    filter_result = final_state.get("evidence_filter_result", {})
    if filter_result:
        print(f"\n📊 Evidence Filtering:")
        print(f"   • Original count: {len(final_state.get('evidence', []))}")
        print(f"   • Filtered count: {len(filtered_evidence)}")
        print(f"   • Removed: {filter_result.get('removed_count', 0)}")
        print(f"   • Quality score: {filter_result.get('quality_score', 0):.2f}")
    
    print("\n" + "="*70)
    print("🧠 FINAL CLINICAL RESPONSE")
    print("="*70 + "\n")
    print(final_state.get("final_answer", "No response generated"))
    
    # Refinement info
    refinement = final_state.get("refinement_result", {})
    if refinement:
        print(f"\n🔧 Refinements Applied: {refinement.get('iterations', 0)}")
        if refinement.get("refinements_applied"):
            print(f"   • Stages: {', '.join(refinement['refinements_applied'])}")
        print(f"   • Final quality: {refinement.get('final_quality_score', 0):.2f}")
    
    print("\n" + "="*70)
    print("📈 EVALUATION METRICS")
    print("="*70)
    
    metrics = final_state.get("metrics", {})
    if metrics:
        for k, v in metrics.items():
            print(f"   • {k}: {v}")
    else:
        print("   No metrics available")
    
    print("\n" + "="*70)
    print("✨ Pipeline completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()