# tests/test_pathology_integration.py

"""
End-to-end integration test for pathology detection
Tests the complete pipeline with pathology detection enabled
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph


def test_pathology_integration():
    """Test complete pipeline with pathology detection"""
    
    print("\n" + "="*80)
    print("PATHOLOGY DETECTION INTEGRATION TEST")
    print("="*80)
    
    # Build graph
    graph = build_mmrag_graph()
    
    # Test case: patient with pleural effusion
    initial_state = {
        "patient_id": 32,  # Known to have effusion
        "query": "are there any signs of pleural effusion",
        "user_role": "doctor",
        "modalities": ["XRAY"],
        "xray_results": [],
        "evidence": [],
        "filtered_evidence": [],
        "evidence_filter_result": {},
        "evidence_gate_result": {},
        "retrieval_attempts": 0,
        "final_answer": "",
        "metrics": {},
        "response_gate_result": {},
        "reasoning_attempts": 0,
        "refinement_count": 0,
        "total_iterations": 0,
        "quality_scores": {}
    }
    
    print("\n[TEST] Running pipeline for patient 32...")
    print(f"[TEST] Query: {initial_state['query']}")
    
    # Run pipeline
    final_state = graph.invoke(initial_state)
    
    # Check results
    print("\n[RESULTS] Pipeline completed")
    print(f"  Iterations: {final_state['total_iterations']}")
    print(f"  Evidence count: {len(final_state['filtered_evidence'])}")
    
    # Check pathology detection
    has_pathology = False
    for e in final_state['filtered_evidence']:
        if "pathology_scores" in e:
            has_pathology = True
            print(f"\n[PATHOLOGY] Evidence {e.get('uid', 'unknown')}:")
            
            if "top_pathologies" in e:
                for pathology, score in e["top_pathologies"]:
                    print(f"    - {pathology}: {score*100:.1f}%")
    
    if has_pathology:
        print("\n✅ Pathology detection is working!")
    else:
        print("\n⚠️  Warning: No pathology scores found in evidence")
    
    # Check if final answer references pathology detection
    final_answer = final_state.get("final_answer", "")
    
    if "effusion" in final_answer.lower() or "pathology" in final_answer.lower():
        print("✅ Final answer incorporates pathology findings")
    else:
        print("⚠️  Final answer may not be using pathology detection")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    
    return final_state


if __name__ == "__main__":
    test_pathology_integration()