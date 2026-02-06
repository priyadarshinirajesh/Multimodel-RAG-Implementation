# test.py

import pandas as pd
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from datetime import datetime

INPUT_FILE = "trial.xlsx"
OUTPUT_FILE = "mmrag_evaluation_results.xlsx"


def run_batch_evaluation():
    print("\n🧪 Running Batch Evaluation for MM-RAG (Verification Enabled)")
    print("=" * 80)

    # Load input Excel
    df = pd.read_excel(INPUT_FILE)
    has_ground_truth = "final_answer" in df.columns

    print(f"📊 Total queries: {len(df)}")
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    if has_ground_truth:
        print("✅ Ground truth column detected")
    print("=" * 80)

    graph = build_mmrag_graph()
    results = []

    start_time = datetime.now()

    for idx, row in df.iterrows():
        patient_id = int(row["patient_id"])
        query = str(row["query"])
        ground_truth = row.get("final_answer") if has_ground_truth else None

        print(f"\n▶ Processing {idx + 1}/{len(df)} | Patient {patient_id}")
        print(f"📝 Query: {query[:80]}")

        # ------------------------------------------------------------
        # ✅ UPDATED: Removed user_role (RBAC removed)
        # ------------------------------------------------------------
        initial_state = {
            "patient_id": patient_id,
            "query": query,

            # XRAY-only execution
            "modalities": ["XRAY"],

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

            # Global tracking
            "total_iterations": 0,
            "quality_scores": {}
        }

        try:
            final_state = graph.invoke(initial_state)

            metrics = final_state.get("metrics", {})
            quality_scores = final_state.get("quality_scores", {})

            evidence_gate = final_state.get("evidence_gate_result", {})
            response_gate = final_state.get("response_gate_result", {})

            evidence_count = len(final_state.get("filtered_evidence", []))

            # Overall quality
            overall_quality = (
                sum(quality_scores.values()) / len(quality_scores)
                if quality_scores else 0
            )

            result_row = {
                # Identification
                "patient_id": patient_id,
                "query": query,
                "status": "SUCCESS",

                # Evaluation metrics
                "Precision@K": metrics.get("Precision@K"),
                "Recall@K": metrics.get("Recall@K"),
                "MRR": metrics.get("MRR"),
                "Groundedness": metrics.get("Groundedness"),
                "ClinicalCorrectness": metrics.get("ClinicalCorrectness"),
                "Completeness": metrics.get("Completeness"),

                # Answers
                "generated_answer": final_state.get("final_answer"),
                "expected_answer": ground_truth,

                # Execution stats
                "total_iterations": final_state.get("total_iterations"),
                "retrieval_attempts": final_state.get("retrieval_attempts"),
                "reasoning_attempts": final_state.get("reasoning_attempts"),
                "refinement_count": final_state.get("refinement_count"),

                # Quality scores
                "evidence_quality": quality_scores.get("evidence"),
                "response_quality": quality_scores.get("response"),
                "overall_quality": overall_quality,

                # Gate decisions
                "evidence_decision": evidence_gate.get("decision", "N/A"),
                "response_decision": response_gate.get("decision", "N/A"),

                # Evidence stats
                "evidence_count": evidence_count,
                "modality": "XRAY"
            }

            results.append(result_row)

            print(f"✅ SUCCESS | Evidence={evidence_count} | "
                  f"EQ={result_row['evidence_quality']:.2f} | "
                  f"RQ={result_row['response_quality']:.2f}")

        except Exception as e:
            print(f"❌ ERROR | Patient {patient_id} | {str(e)}")

            results.append({
                "patient_id": patient_id,
                "query": query,
                "status": "ERROR",
                "generated_answer": str(e),
                "expected_answer": ground_truth
            })

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------
    output_df = pd.DataFrame(results)

    column_order = [
        "patient_id", "query", "status",

        "Precision@K", "Recall@K", "MRR",
        "Groundedness", "ClinicalCorrectness", "Completeness",

        "generated_answer", "expected_answer",

        "total_iterations", "retrieval_attempts",
        "reasoning_attempts", "refinement_count",

        "evidence_quality", "response_quality", "overall_quality",
        "evidence_decision", "response_decision",

        "evidence_count", "modality"
    ]

    output_df = output_df.reindex(columns=column_order)
    output_df.to_excel(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    success_df = output_df[output_df["status"] == "SUCCESS"]

    print("\n" + "=" * 80)
    print("✅ BATCH EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total queries: {len(output_df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Errors: {len(output_df) - len(success_df)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time/query: {elapsed / len(output_df):.2f}s")

    if len(success_df) > 0:
        print("\n📊 AVERAGE METRICS:")
        print(f"  Precision@K: {success_df['Precision@K'].mean():.3f}")
        print(f"  Recall@K: {success_df['Recall@K'].mean():.3f}")
        print(f"  MRR: {success_df['MRR'].mean():.3f}")
        print(f"  Groundedness: {success_df['Groundedness'].mean():.3f}")
        print(f"  Clinical Correctness: {success_df['ClinicalCorrectness'].mean():.3f}")
        print(f"  Completeness: {success_df['Completeness'].mean():.3f}")

        print("\n🎯 QUALITY GATES:")
        print(f"  Evidence PASS: {(success_df['evidence_decision'] == 'PASS').mean()*100:.1f}%")
        print(f"  Response PASS: {(success_df['response_decision'] == 'PASS').mean()*100:.1f}%")

        print("\n🔍 EVIDENCE:")
        print(f"  Avg evidence count: {success_df['evidence_count'].mean():.2f}")

    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation()