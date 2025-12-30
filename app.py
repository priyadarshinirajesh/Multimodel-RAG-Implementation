# app.py (UPDATED)

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph


def main():
    print("\n🧠 Multimodal Clinical Decision Support System")
    print("=" * 50)

    patient_id = int(input("Enter Patient ID: "))
    query = input("Enter Clinical Query: ")

    graph = build_mmrag_graph()

    initial_state = {
        "patient_id": patient_id,
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

    print("\n🔎 RETRIEVED EVIDENCE:")
    for e in final_state["evidence"]:
        print(
            f"- [{e['modality']}] {e['report_text'][:200]}...\n"
        )

    print("\n🧠 FINAL CLINICAL RESPONSE:")
    print(final_state["final_answer"])

    print("\n📊 EVALUATION METRICS")
    print("=" * 40)
    for k, v in final_state["metrics"].items():
        print(f"{k}: {v}")
    print("=" * 40)

    # Display verification results
    print("\n✅ VERIFICATION RESULTS")
    print("=" * 60)
    
    verification = final_state.get("verification_result", {})
    
    if verification:
        print(f"Overall Pass: {verification.get('overall_pass', 'N/A')}")
        print(f"Confidence Score: {verification.get('confidence_score', 0):.2%}")
        print(f"Reruns Performed: {final_state.get('rerun_count', 0)}")
        
        print("\nComponent Scores:")
        print(f"  • Modality Routing: {verification.get('modality_routing', {}).get('score', 0):.2f}")
        print(f"  • Evidence Quality: {verification.get('evidence_quality', {}).get('score', 0):.2f}")
        print(f"  • Clinical Response: {verification.get('clinical_response', {}).get('score', 0):.2f}")
        print(f"  • Citation Check: {verification.get('citation_check', {}).get('score', 0):.2f}")
        
        suggestions = final_state.get("improvement_suggestions", [])
        if suggestions:
            print(f"\n⚠️  Improvement Suggestions Applied ({len(suggestions)}):")
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. [{s['priority']}] {s['agent']}: {s['details']}")
        else:
            print("\n✓ No improvements needed - pipeline executed correctly")
    
    print("=" * 60)


if __name__ == "__main__":
    main()