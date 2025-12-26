# test.py

import pandas as pd
from agents.langgraph_flow.mmrag_graph import build_mmrag_graph

INPUT_FILE = "prompt_queries_150.xlsx"
OUTPUT_FILE = "mmrag_evaluation_results.xlsx"


def run_batch_evaluation():
    print("\n🧪 Running Batch Evaluation for MMRAG")
    print("=" * 60)

    # Load input Excel
    df = pd.read_excel(INPUT_FILE)

    graph = build_mmrag_graph()

    results = []

    for idx, row in df.iterrows():
        patient_id = int(row["patient_id"])
        query = str(row["query"])

        print(f"\n▶ Processing Patient {patient_id}: {query}")

        initial_state = {
            "patient_id": patient_id,
            "query": query,
            "modalities": [],
            "xray_results": [],
            "ct_results": [],
            "mri_results": [],
            "evidence": [],
            "final_answer": "",
            "metrics": {}
        }

        try:
            final_state = graph.invoke(initial_state)

            metrics = final_state.get("metrics", {})

            results.append({
                "patient_id": patient_id,
                "query": query,
                "final_answer": final_state.get("final_answer", ""),
                "Precision@K": metrics.get("Precision@K"),
                "Recall@K": metrics.get("Recall@K"),
                "MRR": metrics.get("MRR"),
                "Groundedness": metrics.get("Groundedness"),
                "ClinicalCorrectness": metrics.get("ClinicalCorrectness"),
                "Completeness": metrics.get("Completeness")
            })

        except Exception as e:
            print(f"❌ Error for patient {patient_id}: {e}")
            results.append({
                "patient_id": patient_id,
                "query": query,
                "final_answer": "ERROR",
                "Precision@K": None,
                "Recall@K": None,
                "MRR": None,
                "Groundedness": None,
                "ClinicalCorrectness": None,
                "Completeness": None
            })

    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n✅ Batch evaluation completed")
    print(f"📄 Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_evaluation()
