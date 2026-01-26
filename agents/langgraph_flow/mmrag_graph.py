# agents/langgraph_flow/mmrag_graph.py

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

# =========================
# VERIFIERS
# =========================
from agents.verifiers.xray_retrieval_contract import verify_xray_retrieval
from agents.verifiers.clinical_safety_validator import validate_clinical_safety
from agents.verifiers.xray_anatomy_validator import validate_xray_anatomy
from agents.verifiers.evidence_consistency_checker import check_evidence_consistency
from agents.verifiers.structure_validator import validate_structure

# =========================
# CORE AGENTS
# =========================
from agents.xray_agent import xray_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent

# =========================
# QUALITY / REFINEMENT
# =========================
from agents.verifiers.evidence_quality_verifier import EvidenceQualityVerifier
from agents.verifiers.response_refiner import ResponseRefiner
from agents.quality_gates import EvidenceQualityGate, ResponseQualityGate


class MMRAgState(TypedDict):
    patient_id: int
    query: str

    xray_results: List[Any]
    evidence: List[Any]
    filtered_evidence: List[Any]

    retrieval_attempts: int
    reasoning_attempts: int
    refinement_count: int

    final_answer: str
    metrics: dict

    evidence_filter_result: dict
    evidence_gate_result: dict
    response_gate_result: dict

    retrieval_contract_result: dict
    consistency_result: dict
    structure_result: dict
    safety_result: dict
    anatomy_result: dict

    total_iterations: int
    quality_scores: dict


# =========================
# NODES
# =========================

def xray_node(state):
    return {"xray_results": xray_agent(state["patient_id"], state["query"])}


def xray_contract_node(state):
    return {"retrieval_contract_result": verify_xray_retrieval(state["xray_results"])}


def aggregation_node(state):
    evidence = aggregate_evidence(state["xray_results"], allowed_modalities=["XRAY"])
    return {
        "evidence": evidence,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }


def evidence_quality_filter_node(state):
    verifier = EvidenceQualityVerifier()
    result = verifier.verify_and_filter(
        query=state["query"],
        evidence=state["evidence"],
        selected_modalities=["XRAY"]
    )
    return {
        "filtered_evidence": result["filtered_evidence"],
        "evidence_filter_result": result
    }


def evidence_quality_gate_node(state):
    gate = EvidenceQualityGate()
    return {
        "evidence_gate_result": gate.evaluate(
            evidence=state["evidence"],
            filter_result=state["evidence_filter_result"],
            query=state["query"]
        )
    }


def evidence_consistency_node(state):
    return {
        "consistency_result": check_evidence_consistency(
            response="",
            evidence=state["filtered_evidence"]
        )
    }


def reasoning_node(state):
    result = clinical_reasoning_agent(
        state["query"],
        state["filtered_evidence"]
    )
    return {
        "final_answer": result["final_answer"],
        "metrics": result["metrics"],
        "reasoning_attempts": state.get("reasoning_attempts", 0) + 1
    }


def clinical_safety_node(state):
    return {"safety_result": validate_clinical_safety(state["final_answer"])}


def xray_anatomy_node(state):
    return {"anatomy_result": validate_xray_anatomy(state["final_answer"])}


def response_refinement_node(state):
    refiner = ResponseRefiner()
    result = refiner.refine_response(
        initial_response=state["final_answer"],
        evidence=state["filtered_evidence"],
        query=state["query"]
    )
    return {
        "final_answer": result["refined_response"],
        "refinement_count": state.get("refinement_count", 0) + 1
    }


def structure_check_node(state):
    return {"structure_result": validate_structure(state["final_answer"])}


def response_quality_gate_node(state):
    gate = ResponseQualityGate()
    return {
        "response_gate_result": gate.evaluate(
            response=state["final_answer"],
            evidence=state["filtered_evidence"],
            metrics=state["metrics"]
        )
    }


def summary_node(state):
    return {
        "quality_scores": {
            "evidence": state["evidence_gate_result"]["score"],
            "response": state["response_gate_result"]["score"]
        },
        "total_iterations": (
            state.get("retrieval_attempts", 0)
            + state.get("reasoning_attempts", 0)
        )
    }


# =========================
# DECISIONS
# =========================

def should_retry_xray(state):
    if state["retrieval_contract_result"]["passed"]:
        return "proceed"
    if state.get("retrieval_attempts", 0) >= 2:
        return "proceed"
    return "retry"

def should_retry_retrieval(state):
    attempts = state.get("retrieval_attempts", 0)

    print(f"[DEBUG] Retrieval attempt {attempts}, decision={state['evidence_gate_result']['decision']}")
    # Allow max 2 retrieval retries
    if state["evidence_gate_result"]["decision"] == "PASS":
        return "proceed"

    if attempts >= 2:
        # Force proceed even if quality is low
        return "proceed"

    return "retry"

def should_refine_for_safety(state):
    return "refine" if not state["safety_result"]["passed"] else "proceed"


def should_refine_for_structure(state):
    return "refine" if not state["structure_result"]["passed"] else "proceed"


def should_refine_response(state):
    if state["response_gate_result"]["decision"] == "PASS":
        return "complete"
    if state.get("refinement_count", 0) >= 2:
        return "complete"
    return "refine"


# =========================
# BUILD GRAPH
# =========================

def build_mmrag_graph():
    graph = StateGraph(MMRAgState)

    # =========================================================
    # REGISTER ALL NODES (MANDATORY)
    # =========================================================
    graph.add_node("xray", xray_node)
    graph.add_node("xray_contract", xray_contract_node)

    graph.add_node("aggregate", aggregation_node)
    graph.add_node("evidence_filter", evidence_quality_filter_node)
    graph.add_node("evidence_gate", evidence_quality_gate_node)
    graph.add_node("evidence_consistency", evidence_consistency_node)

    graph.add_node("reason", reasoning_node)

    graph.add_node("clinical_safety", clinical_safety_node)
    graph.add_node("response_refine", response_refinement_node)
    graph.add_node("structure_check", structure_check_node)
    graph.add_node("response_gate", response_quality_gate_node)

    graph.add_node("summary", summary_node)

    # =========================================================
    # ENTRY POINT
    # =========================================================
    graph.set_entry_point("xray")

    # =========================================================
    # RETRIEVAL FLOW
    # =========================================================
    graph.add_edge("xray", "xray_contract")

    graph.add_conditional_edges(
        "xray_contract",
        should_retry_xray,
        {
            "retry": "xray",
            "proceed": "aggregate"
        }
    )

    graph.add_edge("aggregate", "evidence_filter")
    graph.add_edge("evidence_filter", "evidence_gate")

    graph.add_conditional_edges(
        "evidence_gate",
        should_retry_retrieval,
        {
            "retry": "xray",
            "proceed": "evidence_consistency"
        }
    )

    # =========================================================
    # REASONING
    # =========================================================
    graph.add_edge("evidence_consistency", "reason")

    # =========================================================
    # SAFETY → REFINEMENT → STRUCTURE
    # =========================================================
    graph.add_edge("reason", "clinical_safety")

    graph.add_conditional_edges(
        "clinical_safety",
        should_refine_for_safety,
        {
            "refine": "response_refine",
            "proceed": "response_refine"  # safety ALWAYS refines once
        }
    )

    graph.add_edge("response_refine", "structure_check")

    graph.add_conditional_edges(
        "structure_check",
        should_refine_for_structure,
        {
            "refine": "response_refine",
            "proceed": "response_gate"
        }
    )

    # =========================================================
    # FINAL QUALITY GATE
    # =========================================================
    graph.add_conditional_edges(
        "response_gate",
        should_refine_response,
        {
            "refine": "response_refine",
            "complete": "summary"
        }
    )

    # =========================================================
    # TERMINATION
    # =========================================================
    graph.add_edge("summary", END)

    return graph.compile()
