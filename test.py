# test.py

import pandas as pd
import numpy as np
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from datetime import datetime

INPUT_FILE = "trial.xlsx"
OUTPUT_FILE = "mmrag_evaluation_results.xlsx"


def run_batch_evaluation():
    print("\n🧪 Running Batch Evaluation for MMRAG with Quality Gates")
    print("=" * 80)

    # Load input Excel
    df = pd.read_excel(INPUT_FILE)
    
    # Check if file has expected_final_answer column for comparison
    has_ground_truth = 'final_answer' in df.columns
    
    print(f"📊 Total queries to process: {len(df)}")
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    if has_ground_truth:
        print(f"✅ Ground truth final_answer column found - will include comparison")
    print("=" * 80)

    graph = build_mmrag_graph()

    results = []
    
    start_time = datetime.now()

    for idx, row in df.iterrows():
        patient_id = int(row["patient_id"])
        query = str(row["query"])
        
        # Get ground truth if available
        ground_truth_answer = row.get("final_answer", None) if has_ground_truth else None

        print(f"\n{'='*80}")
        print(f"▶ Query {idx + 1}/{len(df)} | Patient ID: {patient_id}")
        print(f"📝 Query: {query[:70]}{'...' if len(query) > 70 else ''}")
        print(f"{'='*80}")

        initial_state = {
            "patient_id": patient_id,
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
            "refinement_count": 0,
            
            # Global
            "total_iterations": 0,
            "quality_scores": {}
        }

        try:
            final_state = graph.invoke(initial_state)

            metrics = final_state.get("metrics", {})
            quality_scores = final_state.get("quality_scores", {})
            routing_gate = final_state.get("routing_gate_result", {})
            evidence_gate = final_state.get("evidence_gate_result", {})
            response_gate = final_state.get("response_gate_result", {})
            
            # Calculate overall quality
            avg_quality = (
                sum(quality_scores.values()) / len(quality_scores) 
                if quality_scores else 0
            )
            
            # Extract modalities
            selected_modalities = ", ".join(final_state.get("modalities", []))
            
            # Extract evidence count
            evidence_count = len(final_state.get("filtered_evidence", []))
            
            # Prepare result row
            result_row = {
                # Basic info
                "patient_id": patient_id,
                "query": query,
                "generated_answer": final_state.get("final_answer", ""),
                
                # Ground truth comparison (if available)
                "expected_answer": ground_truth_answer if has_ground_truth else None,
                
                # Core Evaluation Metrics (HIGHLIGHTED - from your Excel)
                "Precision@K": metrics.get("Precision@K", None),
                "Recall@K": metrics.get("Recall@K", None),
                "MRR": metrics.get("MRR", None),
                "Groundedness": metrics.get("Groundedness", None),
                "ClinicalCorrectness": metrics.get("ClinicalCorrectness", None),
                "Completeness": metrics.get("Completeness", None),
                
                # Pipeline execution summary
                "total_iterations": final_state.get("total_iterations", 0),
                "routing_attempts": final_state.get("routing_attempts", 0),
                "retrieval_attempts": final_state.get("retrieval_attempts", 0),
                "reasoning_attempts": final_state.get("reasoning_attempts", 0),
                "refinement_count": final_state.get("refinement_count", 0),
                
                # Quality gate scores
                "routing_quality": quality_scores.get("routing", 0),
                "evidence_quality": quality_scores.get("evidence", 0),
                "response_quality": quality_scores.get("response", 0),
                "overall_quality": avg_quality,
                
                # Quality gate decisions
                "routing_decision": routing_gate.get("decision", "N/A"),
                "evidence_decision": evidence_gate.get("decision", "N/A"),
                "response_decision": response_gate.get("decision", "N/A"),
                
                # Selected modalities and evidence
                "selected_modalities": selected_modalities,
                "evidence_count": evidence_count,
                
                # Status
                "status": "SUCCESS"
            }
            
            results.append(result_row)
            
            # Print summary
            print(f"✅ Status: SUCCESS")
            print(f"🔄 Iterations: {result_row['total_iterations']}")
            print(f"📊 Quality: Routing={result_row['routing_quality']:.2f} | "
                  f"Evidence={result_row['evidence_quality']:.2f} | "
                  f"Response={result_row['response_quality']:.2f} | "
                  f"Overall={result_row['overall_quality']:.2f}")
            print(f"🔍 Evidence: {evidence_count} items")
            print(f"🧪 Metrics: P@K={metrics.get('Precision@K', 0):.3f} | "
                  f"R@K={metrics.get('Recall@K', 0):.3f} | "
                  f"Ground={metrics.get('Groundedness', 0):.3f}")

        except Exception as e:
            print(f"❌ Error for patient {patient_id}: {str(e)}")
            
            # Create error row with all fields
            error_row = {
                "patient_id": patient_id,
                "query": query,
                "generated_answer": f"ERROR: {str(e)}",
                
                # Ground truth comparison
                "expected_answer": ground_truth_answer if has_ground_truth else None,
                
                # Core Evaluation Metrics (NULL for errors)
                "Precision@K": None,
                "Recall@K": None,
                "MRR": None,
                "Groundedness": None,
                "ClinicalCorrectness": None,
                "Completeness": None,
                
                # Pipeline execution summary
                "total_iterations": 0,
                "routing_attempts": 0,
                "retrieval_attempts": 0,
                "reasoning_attempts": 0,
                "refinement_count": 0,
                
                # Quality gate scores
                "routing_quality": None,
                "evidence_quality": None,
                "response_quality": None,
                "overall_quality": None,
                
                # Quality gate decisions
                "routing_decision": "ERROR",
                "evidence_decision": "ERROR",
                "response_decision": "ERROR",
                
                # Selected modalities and evidence
                "selected_modalities": None,
                "evidence_count": 0,
                
                # Status
                "status": "ERROR"
            }
            
            results.append(error_row)

    # Calculate total time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Save results to Excel
    output_df = pd.DataFrame(results)
    
    # Reorder columns for better readability (METRICS FIRST like your Excel)
    column_order = [
        # Basic info
        "patient_id", "query", "status",
        
        # ⭐ CORE EVALUATION METRICS (PROMINENT - matching your Excel columns)
        "Precision@K", "Recall@K", "MRR", 
        "Groundedness", "ClinicalCorrectness", "Completeness",
        
        # Answers
        "generated_answer", "expected_answer",
        
        # Pipeline summary
        "total_iterations", "routing_attempts", "retrieval_attempts", 
        "reasoning_attempts", "refinement_count",
        
        # Quality scores
        "routing_quality", "evidence_quality", "response_quality", "overall_quality",
        
        # Quality decisions
        "routing_decision", "evidence_decision", "response_decision",
        
        # Modalities and evidence
        "selected_modalities", "evidence_count"
    ]
    
    output_df = output_df[column_order]
    
    # Save to Excel
    output_df.to_excel(OUTPUT_FILE, index=False)
    
    # ============================================================
    # FINAL SUMMARY STATISTICS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("✅ BATCH EVALUATION COMPLETED")
    print("=" * 80)
    
    # Calculate statistics
    success_count = output_df[output_df['status'] == 'SUCCESS'].shape[0]
    error_count = output_df[output_df['status'] == 'ERROR'].shape[0]
    
    print(f"\n📊 EXECUTION SUMMARY:")
    print(f"  • Total queries processed: {len(output_df)}")
    print(f"  • Successful: {success_count} ({success_count/len(output_df)*100:.1f}%)")
    print(f"  • Errors: {error_count} ({error_count/len(output_df)*100:.1f}%)")
    print(f"  • Total time: {total_time:.2f} seconds")
    print(f"  • Average time per query: {total_time/len(output_df):.2f} seconds")
    
    # Quality gate statistics (only successful runs)
    successful_df = output_df[output_df['status'] == 'SUCCESS']
    
    if len(successful_df) > 0:
        print(f"\n🎯 QUALITY GATE STATISTICS (Successful runs only):")
        
        # Routing quality
        routing_pass = (successful_df['routing_decision'] == 'PASS').sum()
        print(f"  • Routing Gates Passed: {routing_pass}/{len(successful_df)} "
              f"({routing_pass/len(successful_df)*100:.1f}%)")
        print(f"    - Avg Routing Quality: {successful_df['routing_quality'].mean():.3f}")
        
        # Evidence quality
        evidence_pass = (successful_df['evidence_decision'] == 'PASS').sum()
        print(f"  • Evidence Gates Passed: {evidence_pass}/{len(successful_df)} "
              f"({evidence_pass/len(successful_df)*100:.1f}%)")
        print(f"    - Avg Evidence Quality: {successful_df['evidence_quality'].mean():.3f}")
        
        # Response quality
        response_pass = (successful_df['response_decision'] == 'PASS').sum()
        print(f"  • Response Gates Passed: {response_pass}/{len(successful_df)} "
              f"({response_pass/len(successful_df)*100:.1f}%)")
        print(f"    - Avg Response Quality: {successful_df['response_quality'].mean():.3f}")
        
        # Overall quality
        print(f"  • Avg Overall Quality: {successful_df['overall_quality'].mean():.3f}")
        
        print(f"\n🔄 ITERATION STATISTICS:")
        print(f"  • Avg Total Iterations: {successful_df['total_iterations'].mean():.2f}")
        print(f"  • Avg Routing Attempts: {successful_df['routing_attempts'].mean():.2f}")
        print(f"  • Avg Retrieval Attempts: {successful_df['retrieval_attempts'].mean():.2f}")
        print(f"  • Avg Reasoning Attempts: {successful_df['reasoning_attempts'].mean():.2f}")
        print(f"  • Avg Refinement Count: {successful_df['refinement_count'].mean():.2f}")
        
        print(f"\n📈 EVALUATION METRICS (Average):")
        print(f"  • Precision@K: {successful_df['Precision@K'].mean():.3f}")
        print(f"  • Recall@K: {successful_df['Recall@K'].mean():.3f}")
        print(f"  • MRR: {successful_df['MRR'].mean():.3f}")
        print(f"  • Groundedness: {successful_df['Groundedness'].mean():.3f}")
        print(f"  • Clinical Correctness: {successful_df['ClinicalCorrectness'].mean():.3f}")
        print(f"  • Completeness: {successful_df['Completeness'].mean():.3f}")
        
        print(f"\n🔍 EVIDENCE STATISTICS:")
        print(f"  • Avg Evidence Count: {successful_df['evidence_count'].mean():.2f}")
        print(f"  • Min Evidence Count: {successful_df['evidence_count'].min()}")
        print(f"  • Max Evidence Count: {successful_df['evidence_count'].max()}")
        
        # Modality distribution
        print(f"\n🏥 MODALITY DISTRIBUTION:")
        modality_counts = {}
        for modalities in successful_df['selected_modalities']:
            if pd.notna(modalities):
                for mod in modalities.split(", "):
                    modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        for mod, count in sorted(modality_counts.items()):
            print(f"  • {mod}: {count} queries ({count/len(successful_df)*100:.1f}%)")
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    run_batch_evaluation()