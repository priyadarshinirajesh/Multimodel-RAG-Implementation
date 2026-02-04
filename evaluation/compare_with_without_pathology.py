# evaluation/compare_with_without_pathology.py

"""
A/B Testing: Compare system performance with and without pathology detection
"""

import os
import sys
import pandas as pd
import time
import json
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from evaluation.diagnosis_evaluator import DiagnosisEvaluator
from utils.logger import get_logger

logger = get_logger("ComparisonEval")

# Configuration
TEST_DATA_PATH = "data/raw/final_multimodal_dataset.csv"  # Ensure this path is correct
OUTPUT_DIR = "pathology_detection/evaluation/results"
NUM_SAMPLES = 20  # Number of patients to test (keep small for speed, e.g. 20-50)

def run_evaluation_mode(df, use_pathology=True):
    """Runs the pipeline on the dataset with or without pathology detection enabled."""
    results = []
    
    logger.info(f"Starting evaluation with Pathology Detection = {use_pathology}")
    
    # We need to hack the graph or config to disable pathology if needed.
    # For this script, we will simulate "Without" by mocking the detector result if needed,
    # OR simpler: we rely on the graph's internal logic. 
    # NOTE: To strictly test "Without", we can temporarily rename the checkpoint file 
    # or pass a flag if your graph supports it. 
    # For now, we assume the graph ALWAYS uses it if available. 
    # To disable, we will temporarily rename the weights file during the 'False' run.
    
    weights_path = "pathology_detection/weights/best_model.pth"
    temp_path = "pathology_detection/weights/best_model.pth.bak"
    
    if not use_pathology:
        if os.path.exists(weights_path):
            os.rename(weights_path, temp_path)
            logger.info("Temporarily disabled Pathology Model (renamed weights)")
    
    try:
        # Build graph
        graph = build_mmrag_graph()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval (Pathology={use_pathology})"):
            patient_id = int(row['patient_id'])
            query = "What are the findings in the chest X-ray?" # Standard query
            ground_truth = str(row['findings']) + " " + str(row['impression'])
            
            start_time = time.time()
            
            try:
                # Invoke Pipeline
                initial_state = {
                    "patient_id": patient_id,
                    "query": query,
                    "user_role": "doctor",
                    "modalities": ["XRAY"],
                    # Initialize required keys
                    "routing_verification": {}, "routing_gate_result": {},
                    "xray_results": [], "ct_results": [], "mri_results": [],
                    "evidence": [], "filtered_evidence": [],
                    "evidence_filter_result": {}, "evidence_gate_result": {},
                    "retrieval_attempts": 0, "final_answer": "", "metrics": {},
                    "response_gate_result": {}, "refinement_result": {},
                    "reasoning_attempts": 0, "refinement_count": 0,
                    "total_iterations": 0, "quality_scores": {}
                }
                
                response = graph.invoke(initial_state)
                final_answer = response.get("final_answer", "")
                
                duration = time.time() - start_time
                
                results.append({
                    "patient_id": patient_id,
                    "ground_truth": ground_truth,
                    "generated_response": final_answer,
                    "duration": duration,
                    "mode": "With Pathology" if use_pathology else "Without Pathology"
                })
                
            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {e}")
                
    finally:
        # Restore weights if we moved them
        if not use_pathology and os.path.exists(temp_path):
            os.rename(temp_path, weights_path)
            logger.info("Restored Pathology Model weights")

    return results

def calculate_metrics(results):
    evaluator = DiagnosisEvaluator()
    scored_results = []
    
    for r in results:
        scores = evaluator.evaluate(r['generated_response'], r['ground_truth'])
        r.update(scores)
        scored_results.append(r)
        
    return pd.DataFrame(scored_results)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    full_df = pd.read_csv(TEST_DATA_PATH)
    # Filter for unique patients and take a sample
    test_df = full_df.drop_duplicates(subset=['patient_id']).head(NUM_SAMPLES)
    
    print(f"🔬 Running Comparison on {NUM_SAMPLES} patients...")
    
    # 1. Run WITHOUT Pathology
    print("\n[1/2] Running Baseline (Standard RAG)...")
    results_without = run_evaluation_mode(test_df, use_pathology=False)
    
    # 2. Run WITH Pathology
    print("\n[2/2] Running Proposed Method (Pathology-Aware RAG)...")
    results_with = run_evaluation_mode(test_df, use_pathology=True)
    
    # 3. Calculate Metrics
    print("\n[3/3] Calculating BERT/BLEU Scores...")
    df_without = calculate_metrics(results_without)
    df_with = calculate_metrics(results_with)
    
    # 4. Save Results
    df_without.to_csv(f"{OUTPUT_DIR}/results_baseline.csv", index=False)
    df_with.to_csv(f"{OUTPUT_DIR}/results_proposed.csv", index=False)
    
    # 5. Print Summary
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS")
    print("="*50)
    
    metrics = ['bleu_score', 'rouge1', 'bert_similarity', 'clinical_accuracy'] # Adjust based on your evaluator keys
    
    comparison = {}
    for m in metrics:
        if m in df_with.columns:
            val_with = df_with[m].mean()
            val_without = df_without[m].mean()
            imp = ((val_with - val_without) / val_without) * 100 if val_without != 0 else 0
            comparison[m] = (val_without, val_with, imp)
            
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Proposed':<10} | {'Improvement':<10}")
    print("-" * 60)
    for m, (base, prop, imp) in comparison.items():
        print(f"{m:<20} | {base:.4f}     | {prop:.4f}     | {imp:+.2f}%")
    print("="*50)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()