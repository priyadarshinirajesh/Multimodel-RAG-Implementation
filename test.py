# test.py

import pandas as pd
from datetime import datetime

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
from evaluation.diagnosis_evaluator import clinical_correctness

INPUT_FILE  = "trial.xlsx"
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

    print(f"Total queries:          {len(df)}")
    print(f"Input file:             {INPUT_FILE}")
    print(f"Output file:            {OUTPUT_FILE}")
    print(f"Ground truth available: {has_ground_truth}")
    print("=" * 80)

    graph      = build_mmrag_graph()
    results    = []
    start_time = datetime.now()

    for idx, row in df.iterrows():
        patient_id = int(row["patient_id"])
        query      = str(row["query"])

        # Safely extract ground truth answer from Excel
        if has_ground_truth:
            raw = row["final_answer"]
            ground_truth_answer = str(raw).strip() if pd.notna(raw) else ""
            if ground_truth_answer.lower() == "nan":
                ground_truth_answer = ""
        else:
            ground_truth_answer = ""

        expected_answer = ground_truth_answer if ground_truth_answer else None

        print(f"\nProcessing {idx + 1}/{len(df)} | Patient {patient_id}")
        print(f"Query: {query[:120]}")
        print(f"Ground truth: {'provided' if ground_truth_answer else 'missing'}")

        initial_state = {
            "patient_id":  patient_id,
            "query":       query,
            "user_role":   "doctor",

            # ── Required by updated mmrag_graph.py (Phase 1 fix) ──────────────
            # These were missing before and caused KeyErrors in gate nodes
            "evidence_threshold":     0.4,
            "response_threshold":     0.7,
            "max_retrieval_retries":  2,
            "max_refinement_retries": 2,

            # Routing
            "modalities":    ["XRAY"],
            "xray_results":  [],
            "ct_results":    [],
            "mri_results":   [],

            # Evidence
            "evidence":          [],
            "filtered_evidence": [],

            # Counters
            "retrieval_attempts": 0,
            "reasoning_attempts": 0,
            "refinement_count":   0,

            # Required by graph state — was missing before
            "forced_complete": False,

            # Reasoning outputs
            "final_answer": "",
            "metrics":      {},

            # Gate results
            "evidence_filter_result":    {},
            "evidence_gate_result":      {},
            "response_gate_result":      {},

            # Verifier results
            "retrieval_contract_result": {},
            "consistency_result":        {},
            "structure_result":          {},
            "safety_result":             {},
            "anatomy_result":            {},

            # Summary
            "total_iterations": 0,
            "quality_scores":   {},
        }

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": 50}
            )

            metrics        = final_state.get("metrics", {}) or {}
            quality_scores = final_state.get("quality_scores", {}) or {}
            evidence_gate  = final_state.get("evidence_gate_result", {}) or {}
            response_gate  = final_state.get("response_gate_result", {}) or {}

            generated_answer = final_state.get("final_answer", "")
            evidence_count   = len(final_state.get("filtered_evidence", []) or [])

            overall_quality = (
                sum(quality_scores.values()) / len(quality_scores)
                if quality_scores else None
            )

            offline_cc = _offline_correctness(generated_answer, expected_answer)

            result_row = {
                "patient_id": patient_id,
                "query":      query,
                "status":     "SUCCESS",

                # ── Retrieval metrics ──────────────────────────────────────────
                "Precision@K": metrics.get("Precision@K"),
                "Recall@K":    metrics.get("Recall@K"),
                "MRR":         metrics.get("MRR"),

                # ── Response quality metrics ───────────────────────────────────
                "Groundedness":        metrics.get("Groundedness"),
                "GroundednessSource":  metrics.get("GroundednessSource"),
                "ClinicalCorrectness": metrics.get("ClinicalCorrectness"),
                "Completeness":        metrics.get("Completeness"),

                # ── Phase 3 new metrics ────────────────────────────────────────
                "ActionabilityScore": metrics.get("ActionabilityScore"),
                "ContradictionFlag":  metrics.get("ContradictionFlag"),

                # ── Offline evaluation ─────────────────────────────────────────
                "OfflineClinicalCorrectness": offline_cc,

                # ── Generated vs expected ──────────────────────────────────────
                "generated_answer": generated_answer,
                "expected_answer":  expected_answer,

                # ── Pipeline counters ──────────────────────────────────────────
                "total_iterations":   final_state.get("total_iterations"),
                "retrieval_attempts": final_state.get("retrieval_attempts"),
                "reasoning_attempts": final_state.get("reasoning_attempts"),
                "refinement_count":   final_state.get("refinement_count"),
                "forced_complete":    final_state.get("forced_complete", False),

                # ── Quality gate scores ────────────────────────────────────────
                "evidence_quality": quality_scores.get("evidence"),
                "response_quality": quality_scores.get("response"),
                "overall_quality":  overall_quality,

                # ── Gate decisions ─────────────────────────────────────────────
                "evidence_decision": evidence_gate.get("decision", "N/A"),
                "response_decision": response_gate.get("decision", "N/A"),

                # ── Evidence stats ─────────────────────────────────────────────
                "evidence_count": evidence_count,
                "modality":       "XRAY",
            }

            results.append(result_row)

            eq = _safe_float(result_row["evidence_quality"])
            rq = _safe_float(result_row["response_quality"])
            ac = _safe_float(result_row["ActionabilityScore"])
            print(f"SUCCESS | Evidence={evidence_count} | EQ={eq:.2f} | RQ={rq:.2f} | Actionability={ac:.2f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR | Patient {patient_id} | {str(e)}")

            results.append({
                "patient_id": patient_id,
                "query":      query,
                "status":     "ERROR",
                "generated_answer": str(e),
                "expected_answer":  expected_answer,

                "Precision@K":        0.0,
                "Recall@K":           0.0,
                "MRR":                0.0,
                "Groundedness":       0.0,
                "GroundednessSource": "error",
                "ClinicalCorrectness": 0.0,
                "Completeness":       0.0,
                "ActionabilityScore": 0.0,
                "ContradictionFlag":  0.0,
                "OfflineClinicalCorrectness": None,

                "total_iterations":   0,
                "retrieval_attempts": 0,
                "reasoning_attempts": 0,
                "refinement_count":   0,
                "forced_complete":    False,

                "evidence_quality": 0.0,
                "response_quality": 0.0,
                "overall_quality":  0.0,

                "evidence_decision": "ERROR",
                "response_decision": "ERROR",
                "evidence_count":    0,
                "modality":          "XRAY",
            })

    # ── Build output DataFrame ─────────────────────────────────────────────────
    output_df = pd.DataFrame(results)

    preferred_order = [
        "patient_id", "query", "status",
        "Precision@K", "Recall@K", "MRR",
        "Groundedness", "GroundednessSource",
        "ClinicalCorrectness", "Completeness",
        "ActionabilityScore", "ContradictionFlag",
        "OfflineClinicalCorrectness",
        "generated_answer", "expected_answer",
        "total_iterations", "retrieval_attempts",
        "reasoning_attempts", "refinement_count", "forced_complete",
        "evidence_quality", "response_quality", "overall_quality",
        "evidence_decision", "response_decision",
        "evidence_count", "modality",
    ]

    existing_cols  = [c for c in preferred_order if c in output_df.columns]
    remaining_cols = [c for c in output_df.columns if c not in existing_cols]
    output_df      = output_df[existing_cols + remaining_cols]
    output_df.to_excel(OUTPUT_FILE, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    end_time   = datetime.now()
    elapsed    = (end_time - start_time).total_seconds()
    success_df = output_df[output_df["status"] == "SUCCESS"].copy()

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total queries: {len(output_df)}")
    print(f"Successful:    {len(success_df)}")
    print(f"Errors:        {len(output_df) - len(success_df)}")
    print(f"Total time:    {elapsed:.2f}s")
    print(f"Avg/query:     {elapsed / max(len(output_df), 1):.2f}s")

    if len(success_df) > 0:
        print("\nAVERAGE METRICS:")
        for col in [
            "Precision@K", "Recall@K", "MRR",
            "Groundedness", "ClinicalCorrectness", "Completeness",
            "ActionabilityScore", "ContradictionFlag",
            "OfflineClinicalCorrectness",
            "evidence_quality", "response_quality", "overall_quality",
            "evidence_count",
        ]:
            if col in success_df.columns:
                vals = pd.to_numeric(success_df[col], errors="coerce")
                if vals.notna().any():
                    print(f"  {col}: {vals.mean():.3f}")

        if "evidence_decision" in success_df.columns:
            print(f"  Evidence PASS: {(success_df['evidence_decision'] == 'PASS').mean() * 100:.1f}%")
        if "response_decision" in success_df.columns:
            print(f"  Response PASS: {(success_df['response_decision'] == 'PASS').mean() * 100:.1f}%")
        if "forced_complete" in success_df.columns:
            forced = success_df["forced_complete"].sum()
            print(f"  Force-finalized responses: {forced}/{len(success_df)}")

        if "GroundednessSource" in success_df.columns:
            fail_rate  = (success_df["GroundednessSource"] == "ragas_failed").mean() * 100
            ragas_rate = (success_df["GroundednessSource"] == "ragas").mean() * 100
            print(f"  Groundedness via RAGAS: {ragas_rate:.1f}%")
            print(f"  RAGAS failed:           {fail_rate:.1f}%")

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation()