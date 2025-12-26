# agents/langgraph_flow/mmrag_graph.py

from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END

from agents.modality_router_agent import route_modalities
from agents.xray_agent import xray_agent
from agents.ct_agent import ct_agent
from agents.mri_agent import mri_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent


class MMRAgState(TypedDict):
    patient_id: int
    query: str

    # router output
    modalities: List[str]

    # modality agents outputs
    xray_results: List[Any]
    ct_results: List[Any]
    mri_results: List[Any]

    # aggregation + reasoning
    evidence: List[Any]
    final_answer: str
    metrics: dict

def router_node(state):
    print("[INFO] [RouterNode] Routing modalities")
    modalities = route_modalities(state["query"])
    print(f"[DEBUG] Modalities selected: {modalities}")
    return {"modalities": modalities}


def xray_node(state: MMRAgState):
    if "XRAY" in state["modalities"]:
        return {
            "xray_results": xray_agent(
                state["patient_id"],
                state["query"]
            )
        }
    return {"xray_results": []}


def ct_node(state: MMRAgState):
    if "CT" in state["modalities"]:
        return {
            "ct_results": ct_agent(
                state["patient_id"],
                state["query"]
            )
        }
    return {"ct_results": []}


def mri_node(state: MMRAgState):
    if "MRI" in state["modalities"]:
        return {
            "mri_results": mri_agent(
                state["patient_id"],
                state["query"]
            )
        }
    return {"mri_results": []}


def precision_recall_mrr(retrieved, ground_truth, k=7):
    relevant = set(ground_truth)
    retrieved_k = retrieved[:k]
    retrieved_set = set(retrieved_k)

    true_positives = len(retrieved_set & relevant)
    retrieved_count = len(retrieved_set)

    precision = (
        true_positives / retrieved_count
        if retrieved_count > 0 else 0
    )

    recall = (
        true_positives / len(relevant)
        if relevant else 0
    )

    mrr = 0
    for idx, item in enumerate(retrieved_k, start=1):
        if item in relevant:
            mrr = 1 / idx
            break

    return {
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "MRR": round(mrr, 3)
    }
def add_distractors(evidence, k=2):
    distractors = [
        {
            "modality": "XRAY",
            "report_text": "Normal chest X-ray. No acute findings.",
            "image_path": None,
            "has_image": False
        },
        {
            "modality": "CT",
            "report_text": "Abdominal CT shows normal liver and spleen.",
            "image_path": None,
            "has_image": False
        }
    ]
    return evidence + distractors[:k]

def aggregation_node(state):
    evidence = aggregate_evidence(
        state["xray_results"]
        + state["ct_results"]
        + state["mri_results"]
    )

    evidence = add_distractors(evidence, k=2)

    return {"evidence": evidence}



def reasoning_node(state):
    print("[INFO] [ReasoningNode] Invoking DeepSeek with multimodal evidence")

    result = clinical_reasoning_agent(
        state["query"],
        state["evidence"]
    )

    return {
        "final_answer": result["final_answer"],
        "metrics": result["metrics"]
    }



def build_mmrag_graph():
    graph = StateGraph(MMRAgState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("xray", xray_node)
    graph.add_node("ct", ct_node)
    graph.add_node("mri", mri_node)
    graph.add_node("aggregate", aggregation_node)
    graph.add_node("reason", reasoning_node)

    # Define execution flow
    graph.set_entry_point("router")

    graph.add_edge("router", "xray")
    graph.add_edge("router", "ct")
    graph.add_edge("router", "mri")

    graph.add_edge("xray", "aggregate")
    graph.add_edge("ct", "aggregate")
    graph.add_edge("mri", "aggregate")

    graph.add_edge("aggregate", "reason")
    graph.add_edge("reason", END)

    return graph.compile()
