# app.py

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph
#from scripts.start_llava_med import is_running, start_llava_med

def main():
    print("\n Multimodal Clinical Decision Support System")
    print("=" * 50)

    # if not is_running():
    #     start_llava_med()

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
        "metrics": {}
    }

    final_state = graph.invoke(initial_state)

    print("\n RETRIEVED EVIDENCE:")
    for e in final_state["evidence"]:
        print(
            f"- [{e['modality']}] {e['report_text'][:200]}...\n"
        )

    print("\n FINAL CLINICAL RESPONSE:")
    print(final_state["final_answer"])

    print("\n EVALUATION METRICS")
    print("=" * 40)
    for k, v in final_state["metrics"].items():
        print(f"{k}: {v}")
    print("=" * 40)


if __name__ == "__main__":
    main()
