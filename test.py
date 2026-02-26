# test.py

import pandas as pd
from datetime import datetime

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from evaluation.diagnosis_evaluator import clinical_correctness

INPUT_FILE = "trial.xlsx"
OUTPUT_FILE = "mmrag_evaluation_results.xlsx"


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _offline_correctness(generated_answer, expected_answer):
    """Offline semantic similarity against provided ground-truth answer."""
    if not isinstance(generated_answer, str) or not generated_answer.strip():
        return None
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        return None
    try:
        return float(clinical_correctness(generated_answer, [expected_answer]))
    except Exception:
        return None


def run_batch_evaluation():
    print("\nRunning Batch Evaluation for MM-RAG")
    print("=" * 80)

    df = pd.read_excel(INPUT_FILE)
    has_ground_truth = "final_answer" in df.columns

    print(f"Total queries: {len(df)}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Ground truth available: {has_ground_truth}")
    print("=" * 80)

    graph = build_mmrag_graph()
    results = []
    start_time = datetime.now()

    for idx, row in df.iterrows():
        patient_id = int(row["patient_id"])
        query = str(row["query"])
        expected_answer = row["final_answer"] if has_ground_truth else None

        print(f"\nProcessing {idx + 1}/{len(df)} | Patient {patient_id}")
        print(f"Query: {query[:120]}")

        initial_state = {
            "patient_id": patient_id,
            "query": query,
            "user_role": "doctor",

            "modalities": ["XRAY"],

            "xray_results": [],
            "evidence": [],
            "filtered_evidence": [],

            "retrieval_attempts": 0,
            "reasoning_attempts": 0,
            "refinement_count": 0,

            "final_answer": "",
            "metrics": {},

            "evidence_filter_result": {},
            "evidence_gate_result": {},
            "response_gate_result": {},

            "retrieval_contract_result": {},
            "consistency_result": {},
            "structure_result": {},
            "safety_result": {},
            "anatomy_result": {},

            "total_iterations": 0,
            "quality_scores": {}
        }

        try:
            final_state = graph.invoke(initial_state)

            metrics = final_state.get("metrics", {}) or {}
            quality_scores = final_state.get("quality_scores", {}) or {}

            evidence_gate = final_state.get("evidence_gate_result", {}) or {}
            response_gate = final_state.get("response_gate_result", {}) or {}

            generated_answer = final_state.get("final_answer", "")
            evidence_count = len(final_state.get("filtered_evidence", []) or [])

            overall_quality = (
                sum(quality_scores.values()) / len(quality_scores)
                if quality_scores else None
            )

            offline_cc = _offline_correctness(generated_answer, expected_answer)

            result_row = {
                "patient_id": patient_id,
                "query": query,
                "status": "SUCCESS",

                # Online metrics (from pipeline)
                "Precision@K": metrics.get("Precision@K"),
                "Recall@K": metrics.get("Recall@K"),
                "MRR": metrics.get("MRR"),
                "Groundedness": metrics.get("Groundedness"),
                "GroundednessSource": metrics.get("GroundednessSource"),
                "ClinicalCorrectness": metrics.get("ClinicalCorrectness"),
                "Completeness": metrics.get("Completeness"),

                # Offline metric (against Excel ground truth)
                "OfflineClinicalCorrectness": offline_cc,

                "generated_answer": generated_answer,
                "expected_answer": expected_answer,

                "total_iterations": final_state.get("total_iterations"),
                "retrieval_attempts": final_state.get("retrieval_attempts"),
                "reasoning_attempts": final_state.get("reasoning_attempts"),
                "refinement_count": final_state.get("refinement_count"),

                "evidence_quality": quality_scores.get("evidence"),
                "response_quality": quality_scores.get("response"),
                "overall_quality": overall_quality,

                "evidence_decision": evidence_gate.get("decision", "N/A"),
                "response_decision": response_gate.get("decision", "N/A"),

                "evidence_count": evidence_count,
                "modality": "XRAY"
            }

            results.append(result_row)

            eq = _safe_float(result_row["evidence_quality"])
            rq = _safe_float(result_row["response_quality"])
            print(f"SUCCESS | Evidence={evidence_count} | EQ={eq:.2f} | RQ={rq:.2f}")

        except Exception as e:
            print(f"ERROR | Patient {patient_id} | {str(e)}")
            results.append({
                "patient_id": patient_id,
                "query": query,
                "status": "ERROR",
                "generated_answer": str(e),
                "expected_answer": expected_answer
            })

    output_df = pd.DataFrame(results)

    preferred_order = [
        "patient_id", "query", "status",
        "Precision@K", "Recall@K", "MRR",
        "Groundedness","GroundednessSource", "ClinicalCorrectness", "Completeness", "OfflineClinicalCorrectness",
        "generated_answer", "expected_answer",
        "total_iterations", "retrieval_attempts", "reasoning_attempts", "refinement_count",
        "evidence_quality", "response_quality", "overall_quality",
        "evidence_decision", "response_decision",
        "evidence_count", "modality"
    ]

    existing_cols = [c for c in preferred_order if c in output_df.columns]
    remaining_cols = [c for c in output_df.columns if c not in existing_cols]
    output_df = output_df[existing_cols + remaining_cols]
    output_df.to_excel(OUTPUT_FILE, index=False)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    success_df = output_df[output_df["status"] == "SUCCESS"].copy()

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total queries: {len(output_df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Errors: {len(output_df) - len(success_df)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time/query: {elapsed / max(len(output_df), 1):.2f}s")

    if len(success_df) > 0:
        print("\nAVERAGE METRICS:")
        for col in [
            "Precision@K", "Recall@K", "MRR",
            "Groundedness", "ClinicalCorrectness", "Completeness",
            "OfflineClinicalCorrectness",
            "evidence_quality", "response_quality", "overall_quality",
            "evidence_count"
        ]:
            if col in success_df.columns:
                vals = pd.to_numeric(success_df[col], errors="coerce")
                if vals.notna().any():
                    print(f"  {col}: {vals.mean():.3f}")

        if "evidence_decision" in success_df.columns:
            print(f"  Evidence PASS: {(success_df['evidence_decision'] == 'PASS').mean() * 100:.1f}%")
        if "response_decision" in success_df.columns:
            print(f"  Response PASS: {(success_df['response_decision'] == 'PASS').mean() * 100:.1f}%")
        
        if "GroundednessSource" in success_df.columns:
            fail_rate = (success_df["GroundednessSource"] == "ragas_failed").mean() * 100
            ragas_rate = (success_df["GroundednessSource"] == "ragas").mean() * 100
            print(f"  Groundedness via RAGAS: {ragas_rate:.1f}%")
            print(f"  RAGAS failed: {fail_rate:.1f}%")

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation()
