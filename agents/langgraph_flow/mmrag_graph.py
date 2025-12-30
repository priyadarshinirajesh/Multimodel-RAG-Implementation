# agents/langgraph_flow/mmrag_graph.py (UPDATED)

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from agents.modality_router_agent import route_modalities
from agents.xray_agent import xray_agent
from agents.ct_agent import ct_agent
from agents.mri_agent import mri_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent
from agents.verification_agent import VerificationAgent


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
    
    # verification outputs
    verification_result: dict
    improvement_suggestions: List[dict]
    requires_rerun: bool
    rerun_count: int  # Track number of reruns


def router_node(state):
    print("[INFO] [RouterNode] Routing modalities")
    modalities = route_modalities(state["query"])
    print(f"[DEBUG] Modalities selected: {modalities}")
    return {"modalities": modalities}


def xray_node(state: MMRAgState):
    if "XRAY" in state["modalities"]:
        # Adjust retrieval limit if suggested
        limit = 5
        if state.get("improvement_suggestions"):
            for suggestion in state["improvement_suggestions"]:
                if suggestion["agent"] == "retrieval_agents" and "increase_retrieval_limit" in suggestion["action"]:
                    limit = 7
        
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


def add_distractors(evidence, k=2):
    """Add distractor evidence for robustness testing"""
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


def verification_node(state: MMRAgState):
    """Verification Agent node - validates pipeline execution"""
    
    print("\n[INFO] [VerificationNode] Starting verification")
    
    verifier = VerificationAgent()
    
    verification_output = verifier.verify_pipeline(
        query=state["query"],
        selected_modalities=state["modalities"],
        evidence=state["evidence"],
        final_answer=state["final_answer"],
        metrics=state["metrics"]
    )
    
    return {
        "verification_result": verification_output["verification_result"],
        "improvement_suggestions": verification_output["improvement_suggestions"],
        "requires_rerun": verification_output["requires_rerun"]
    }


def should_rerun(state: MMRAgState) -> str:
    """Conditional edge: decide whether to rerun or end"""
    
    # Prevent infinite loops - max 2 reruns
    rerun_count = state.get("rerun_count", 0)
    if rerun_count >= 2:
        print("[INFO] Max reruns reached (2). Proceeding to END.")
        return "end"
    
    if state.get("requires_rerun", False):
        print(f"[INFO] Rerun required. Iteration: {rerun_count + 1}")
        return "rerun"
    
    return "end"


def rerun_preparation_node(state: MMRAgState):
    """Prepare state for rerun based on improvement suggestions"""
    
    print("\n[INFO] [RerunPreparation] Applying improvements")
    
    suggestions = state.get("improvement_suggestions", [])
    
    # Apply modality routing changes
    for suggestion in suggestions:
        if suggestion["agent"] == "modality_router":
            if suggestion["action"] == "add_modalities":
                # Extract modalities to add from details
                import re
                match = re.search(r'\[(.*?)\]', suggestion["details"])
                if match:
                    new_modalities = eval(match.group(0))
                    current_modalities = set(state["modalities"])
                    current_modalities.update(new_modalities)
                    state["modalities"] = list(current_modalities)
                    print(f"[RerunPrep] Updated modalities: {state['modalities']}")
    
    # Increment rerun counter
    rerun_count = state.get("rerun_count", 0) + 1
    
    return {
        "modalities": state["modalities"],
        "rerun_count": rerun_count,
        # Clear previous results to force re-retrieval
        "xray_results": [],
        "ct_results": [],
        "mri_results": [],
        "evidence": [],
        "final_answer": "",
        "metrics": {}
    }


def build_mmrag_graph():
    """Build the MM-RAG graph with verification loop"""
    
    graph = StateGraph(MMRAgState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("xray", xray_node)
    graph.add_node("ct", ct_node)
    graph.add_node("mri", mri_node)
    graph.add_node("aggregate", aggregation_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("verify", verification_node)
    graph.add_node("rerun_prep", rerun_preparation_node)

    # Define execution flow
    graph.set_entry_point("router")

    # Parallel modality retrieval
    graph.add_edge("router", "xray")
    graph.add_edge("router", "ct")
    graph.add_edge("router", "mri")

    # Converge to aggregation
    graph.add_edge("xray", "aggregate")
    graph.add_edge("ct", "aggregate")
    graph.add_edge("mri", "aggregate")

    # Reasoning -> Verification
    graph.add_edge("aggregate", "reason")
    graph.add_edge("reason", "verify")

    # Conditional edge: rerun or end
    graph.add_conditional_edges(
        "verify",
        should_rerun,
        {
            "rerun": "rerun_prep",
            "end": END
        }
    )

    # Rerun loop back to router
    graph.add_edge("rerun_prep", "router")

    return graph.compile()