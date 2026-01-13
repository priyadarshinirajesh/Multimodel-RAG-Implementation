# agents/langgraph_flow/mmrag_graph.py (COMPLETELY UPDATED)

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from agents.modality_router_agent import route_modalities
from agents.xray_agent import xray_agent
from agents.ct_agent import ct_agent
from agents.mri_agent import mri_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent

# NEW IMPORTS
from agents.verifiers.router_verifier import RouterVerifier
from agents.verifiers.evidence_quality_verifier import EvidenceQualityVerifier
from agents.verifiers.response_refiner import ResponseRefiner
from agents.quality_gates import RoutingQualityGate, EvidenceQualityGate, ResponseQualityGate


class MMRAgState(TypedDict):
    patient_id: int
    query: str

    # Router outputs
    modalities: List[str]
    routing_attempts: int
    routing_verification: dict
    routing_gate_result: dict

    # Retrieval outputs
    xray_results: List[Any]
    ct_results: List[Any]
    mri_results: List[Any]
    
    # Evidence outputs
    evidence: List[Any]
    filtered_evidence: List[Any]
    evidence_filter_result: dict
    evidence_gate_result: dict
    retrieval_attempts: int

    # Reasoning outputs
    final_answer: str
    metrics: dict
    response_gate_result: dict
    refinement_result: dict
    reasoning_attempts: int
    refinement_count: int  # ADD THIS LINE
    
    # Global tracking
    total_iterations: int
    quality_scores: dict


# ============================================================
# ROUTING STAGE WITH LOCAL FEEDBACK
# ============================================================

def router_node(state):
    """Route query to modalities with local retry capability"""
    print("\n[INFO] [RouterNode] Starting modality routing...")
    
    modalities = route_modalities(state["query"])
    
    return {
        "modalities": modalities,
        "routing_attempts": state.get("routing_attempts", 0) + 1
    }


def router_verification_node(state):
    """Verify routing decision immediately"""
    print("[INFO] [RouterVerification] Verifying routing...")
    
    verifier = RouterVerifier()
    verification_result = verifier.verify_routing(
        state["query"],
        state["modalities"]
    )
    
    return {"routing_verification": verification_result}


def routing_quality_gate_node(state):
    """Quality gate for routing"""
    print("[INFO] [RoutingQualityGate] Evaluating...")
    
    gate = RoutingQualityGate()
    gate_result = gate.evaluate(
        query=state["query"],
        selected_modalities=state["modalities"],
        verification_result=state["routing_verification"]
    )
    
    return {"routing_gate_result": gate_result}


def should_retry_routing(state) -> str:
    """Decide if routing needs retry"""
    
    gate_result = state.get("routing_gate_result", {})
    decision = gate_result.get("decision", "PASS")
    attempts = state.get("routing_attempts", 0)
    
    if decision == "PASS":
        print("[INFO] Routing PASSED - proceeding to retrieval")
        return "proceed"
    
    if attempts >= 2:
        print("[WARNING] Max routing attempts reached - proceeding anyway")
        return "proceed"
    
    if decision == "FAIL" or decision == "RETRY":
        print(f"[INFO] Routing {decision} - retrying with corrections")
        return "retry"
    
    return "proceed"


def routing_correction_node(state):
    """Apply corrections to routing based on verification feedback"""
    print("[INFO] [RoutingCorrection] Applying corrections...")
    
    verification = state.get("routing_verification", {})
    suggested_modalities = verification.get("suggested_modalities", state["modalities"])
    
    print(f"[INFO] Correcting modalities: {state['modalities']} → {suggested_modalities}")
    
    return {
        "modalities": suggested_modalities,
        "routing_attempts": state.get("routing_attempts", 0)
    }


# ============================================================
# RETRIEVAL STAGE (No changes needed - already parallel)
# ============================================================

def xray_node(state):
    if "XRAY" in state["modalities"]:
        return {"xray_results": xray_agent(state["patient_id"], state["query"])}
    return {"xray_results": []}


def ct_node(state):
    if "CT" in state["modalities"]:
        return {"ct_results": ct_agent(state["patient_id"], state["query"])}
    return {"ct_results": []}


def mri_node(state):
    if "MRI" in state["modalities"]:
        return {"mri_results": mri_agent(state["patient_id"], state["query"])}
    return {"mri_results": []}


# ============================================================
# EVIDENCE STAGE WITH QUALITY FILTERING
# ============================================================

def aggregation_node(state):
    """Aggregate evidence from all modalities"""
    print("[INFO] [Aggregation] Merging evidence...")
    
    evidence = aggregate_evidence(
        state["xray_results"] + state["ct_results"] + state["mri_results"],
        allowed_modalities=state["modalities"]
    )
    
    return {
        "evidence": evidence,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }


def evidence_quality_filter_node(state):
    """Filter and verify evidence quality"""
    print("[INFO] [EvidenceFilter] Filtering evidence...")
    
    verifier = EvidenceQualityVerifier()
    filter_result = verifier.verify_and_filter(
        query=state["query"],
        evidence=state["evidence"],
        selected_modalities=state["modalities"]
    )
    
    return {
        "filtered_evidence": filter_result["filtered_evidence"],
        "evidence_filter_result": filter_result
    }


def evidence_quality_gate_node(state):
    """Quality gate for evidence"""
    print("[INFO] [EvidenceQualityGate] Evaluating...")
    
    gate = EvidenceQualityGate()
    gate_result = gate.evaluate(
        evidence=state["evidence"],
        filter_result=state["evidence_filter_result"],
        query=state["query"]
    )
    
    return {"evidence_gate_result": gate_result}


def should_retry_retrieval(state) -> str:
    """Decide if retrieval needs retry"""
    
    gate_result = state.get("evidence_gate_result", {})
    decision = gate_result.get("decision", "PASS")
    attempts = state.get("retrieval_attempts", 0)
    
    if decision == "PASS":
        print("[INFO] Evidence quality PASSED - proceeding to reasoning")
        return "proceed"
    
    if attempts >= 2:
        print("[WARNING] Max retrieval attempts reached - proceeding anyway")
        return "proceed"
    
    if decision == "FAIL":
        print("[INFO] Evidence quality FAILED - retrying retrieval")
        return "retry"
    
    return "proceed"


def retrieval_adjustment_node(state):
    """Adjust retrieval parameters based on feedback"""
    print("[INFO] [RetrievalAdjustment] Adjusting retrieval parameters...")
    
    # Could increase limit, change query, etc.
    # For now, just increment attempts to trigger different retrieval
    
    return {
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        # Clear evidence to force re-retrieval
        "evidence": [],
        "filtered_evidence": []
    }


# ============================================================
# REASONING STAGE WITH PROGRESSIVE REFINEMENT
# ============================================================

def reasoning_node(state):
    """Generate clinical response"""
    print("[INFO] [ReasoningNode] Generating clinical response...")
    
    # Use filtered evidence (high quality)
    evidence = state.get("filtered_evidence", state.get("evidence", []))
    
    result = clinical_reasoning_agent(state["query"], evidence)
    
    return {
        "final_answer": result["final_answer"],
        "metrics": result["metrics"],
        "reasoning_attempts": state.get("reasoning_attempts", 0) + 1
    }


def response_quality_gate_node(state):
    """Quality gate for response"""
    print("[INFO] [ResponseQualityGate] Evaluating...")
    
    gate = ResponseQualityGate()
    gate_result = gate.evaluate(
        response=state["final_answer"],
        evidence=state.get("filtered_evidence", []),
        metrics=state["metrics"]
    )
    
    return {"response_gate_result": gate_result}


def should_refine_response(state) -> str:
    """Decide if response needs refinement"""
    
    gate_result = state.get("response_gate_result", {})
    decision = gate_result.get("decision", "PASS")
    attempts = state.get("reasoning_attempts", 0)
    
    if decision == "PASS":
        print("[INFO] Response quality PASSED - completing")
        return "complete"
    
    # CRITICAL FIX: Max 2 refinement attempts (not reasoning attempts)
    refinement_count = state.get("refinement_count", 0)
    if refinement_count >= 2:  # Changed from reasoning_attempts to refinement_count
        print(f"[WARNING] Max refinement attempts reached ({refinement_count}) - completing anyway")
        return "complete"
    
    if decision in ["FAIL", "RETRY"]:
        print(f"[INFO] Response needs refinement (attempt {refinement_count + 1}/2)")
        return "refine"
    
    return "complete"


def response_refinement_node(state):
    """Apply progressive refinement to response"""
    print("[INFO] [ResponseRefinement] Applying progressive refinement...")
    
    refiner = ResponseRefiner()
    refinement_result = refiner.refine_response(
        initial_response=state["final_answer"],
        evidence=state.get("filtered_evidence", []),
        query=state["query"]
    )
    
    # Increment refinement counter
    refinement_count = state.get("refinement_count", 0) + 1
    
    return {
        "final_answer": refinement_result["refined_response"],
        "refinement_result": refinement_result,
        "refinement_count": refinement_count  # ADD THIS LINE
    }


# ============================================================
# FINAL SUMMARY NODE
# ============================================================

def summary_node(state):
    """Generate final summary of quality and iterations"""
    print("\n[INFO] [Summary] Generating pipeline summary...")
    
    quality_scores = {
        "routing": state.get("routing_gate_result", {}).get("score", 0),
        "evidence": state.get("evidence_gate_result", {}).get("score", 0),
        "response": state.get("response_gate_result", {}).get("score", 0)
    }
    
    total_iterations = (
        state.get("routing_attempts", 0) +
        state.get("retrieval_attempts", 0) +
        state.get("reasoning_attempts", 0)
    )
    
    return {
        "quality_scores": quality_scores,
        "total_iterations": total_iterations
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_mmrag_graph():
    """Build the improved MM-RAG graph with quality gates and local feedback"""
    
    graph = StateGraph(MMRAgState)

    # ===== ROUTING STAGE =====
    graph.add_node("router", router_node)
    graph.add_node("router_verification", router_verification_node)
    graph.add_node("routing_quality_gate", routing_quality_gate_node)
    graph.add_node("routing_correction", routing_correction_node)
    
    # ===== RETRIEVAL STAGE =====
    graph.add_node("xray", xray_node)
    graph.add_node("ct", ct_node)
    graph.add_node("mri", mri_node)
    
    # ===== EVIDENCE STAGE =====
    graph.add_node("aggregate", aggregation_node)
    graph.add_node("evidence_filter", evidence_quality_filter_node)
    graph.add_node("evidence_quality_gate", evidence_quality_gate_node)
    graph.add_node("retrieval_adjustment", retrieval_adjustment_node)
    
    # ===== REASONING STAGE =====
    graph.add_node("reason", reasoning_node)
    graph.add_node("response_quality_gate", response_quality_gate_node)
    graph.add_node("response_refinement", response_refinement_node)
    
    # ===== SUMMARY =====
    graph.add_node("summary", summary_node)

    # ===== DEFINE EDGES =====
    
    # Entry point
    graph.set_entry_point("router")
    
    # Routing stage flow
    graph.add_edge("router", "router_verification")
    graph.add_edge("router_verification", "routing_quality_gate")
    
    # Routing decision
    graph.add_conditional_edges(
        "routing_quality_gate",
        should_retry_routing,
        {
            "retry": "routing_correction",
            "proceed": "xray"  # Start parallel retrieval
        }
    )
    
    # Routing correction loops back
    graph.add_edge("routing_correction", "router")
    
    # Parallel retrieval (triggered from routing_quality_gate)
    graph.add_edge("routing_quality_gate", "ct")
    graph.add_edge("routing_quality_gate", "mri")
    
    # Converge to aggregation
    graph.add_edge("xray", "aggregate")
    graph.add_edge("ct", "aggregate")
    graph.add_edge("mri", "aggregate")
    
    # Evidence stage flow
    graph.add_edge("aggregate", "evidence_filter")
    graph.add_edge("evidence_filter", "evidence_quality_gate")
    
    # Evidence decision
    graph.add_conditional_edges(
        "evidence_quality_gate",
        should_retry_retrieval,
        {
            "retry": "retrieval_adjustment",
            "proceed": "reason"
        }
    )
    
    # Retrieval adjustment loops back
    graph.add_edge("retrieval_adjustment", "xray")
    graph.add_edge("retrieval_adjustment", "ct")
    graph.add_edge("retrieval_adjustment", "mri")
    
    # Reasoning stage flow
    graph.add_edge("reason", "response_quality_gate")
    
    # Response decision
    graph.add_conditional_edges(
        "response_quality_gate",
        should_refine_response,
        {
            "refine": "response_refinement",
            "complete": "summary"
        }
    )
    
    # Refinement loops back to quality gate
    graph.add_edge("response_refinement", "response_quality_gate")
    
    # Summary to end
    graph.add_edge("summary", END)

    return graph.compile()
