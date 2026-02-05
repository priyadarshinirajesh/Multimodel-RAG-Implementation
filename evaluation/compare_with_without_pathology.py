# evaluation/compare_with_without_pathology.py
"""
A/B Testing: Compare system performance with and without pathology detection
"""

import pandas as pd
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from tqdm import tqdm
import json


def run_comparison(test_queries_file="trial.xlsx", num_samples=20):
    """
    Run A/B test comparing:
    - Baseline: Without pathology detection
    - Enhanced: With pathology detection
    """
    
    print("="*80)
    print("A/B TESTING: WITH vs WITHOUT PATHOLOGY DETECTION")
    print("="*80)
    
    # Load test queries
    df = pd.read_excel(test_queries_file).head(num_samples)
    
    graph = build_mmrag_graph()
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running tests"):
        patient_id = int(row["patient_id"])
        query = str(row["query"])
        
        # Create initial state
        state = {
            "patient_id": patient_id,
            "query": query,
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
        
        # Run pipeline
        final_state = graph.invoke(state)
        
        # Extract metrics
        metrics = final_state.get("metrics", {})
        
        # Check if pathology detection was used
        has_pathology = any(
            "pathology_scores" in e 
            for e in final_state.get("filtered_evidence", [])
        )
        
        results.append({
            "patient_id": patient_id,
            "query": query,
            "clinical_correctness": metrics.get("ClinicalCorrectness", 0),
            "groundedness": metrics.get("Groundedness", 0),
            "completeness": metrics.get("Completeness", 0),
            "used_pathology_detection": has_pathology,
            "total_iterations": final_state.get("total_iterations", 0),
            "final_answer_length": len(final_state.get("final_answer", ""))
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    avg_clinical_correctness = results_df["clinical_correctness"].mean()
    avg_groundedness = results_df["groundedness"].mean()
    avg_completeness = results_df["completeness"].mean()
    
    print(f"\nAverage Clinical Correctness: {avg_clinical_correctness:.4f}")
    print(f"Average Groundedness:         {avg_groundedness:.4f}")
    print(f"Average Completeness:         {avg_completeness:.4f}")
    print(f"Pathology Detection Used:     {results_df['used_pathology_detection'].sum()}/{len(results_df)}")
    
    # Save results
    results_df.to_csv("pathology_comparison_results.csv", index=False)
    print(f"\n✅ Results saved to: pathology_comparison_results.csv")
    
    return results_df


if __name__ == "__main__":
    run_comparison()