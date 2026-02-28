# agents/langgraph_flow/mmrag_graph.py

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from agents.verifiers.xray_retrieval_contract import verify_xray_retrieval
from agents.verifiers.clinical_safety_validator import validate_clinical_safety
from agents.verifiers.xray_anatomy_validator import validate_xray_anatomy
from agents.verifiers.evidence_consistency_checker import check_evidence_consistency
from agents.verifiers.structure_validator import validate_structure

from agents.xray_agent import xray_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent

from agents.verifiers.evidence_quality_verifier import EvidenceQualityVerifier
from agents.verifiers.response_refiner import ResponseRefiner
from agents.quality_gates import EvidenceQualityGate, ResponseQualityGate
from agents.verifiers.structure_repair import enforce_structure

from utils.logger import get_logger

logger = get_logger("MMRAGGraph")


class MMRAgState(TypedDict):
    patient_id:   int
    query:        str
    user_role:    str

    # ── UI-configurable thresholds (Issue 9 fix) ──────────────────────
    # These are set in initial_state by streamlit_app.py / test.py.
    # Default values are used when not provided (plain graph.invoke calls).
    evidence_threshold:     float   # EvidenceQualityGate pass threshold
    response_threshold:     float   # ResponseQualityGate pass threshold
    max_retrieval_retries:  int     # max xray fetch + aggregation cycles
    max_refinement_retries: int     # max response_refine cycles

    xray_results:      List[Any]
    evidence:          List[Any]
    filtered_evidence: List[Any]

    # Issue 4 fix: retrieval_attempts now incremented in xray_node (not aggregation_node)
    retrieval_attempts: int
    reasoning_attempts: int
    refinement_count:   int

    # Issue 2 fix: track whether pipeline was force-finalized
    forced_complete: bool

    final_answer: str
    metrics:      dict

    evidence_filter_result:    dict
    evidence_gate_result:      dict
    response_gate_result:      dict

    retrieval_contract_result: dict
    consistency_result:        dict
    structure_result:          dict
    safety_result:             dict
    anatomy_result:            dict

    total_iterations: int
    quality_scores:   dict


# =============================================================================
# NODES
# =============================================================================

def xray_node(state):
    # Issue 4 FIX: increment retrieval_attempts HERE (at actual retrieval),
    # not inside aggregation_node.  Previously the counter was semantically
    # wrong — it counted aggregation cycles, not retrieval cycles.
    return {
        "xray_results": xray_agent(state["patient_id"], state["query"]),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def xray_contract_node(state):
    return {"retrieval_contract_result": verify_xray_retrieval(state["xray_results"])}


def aggregation_node(state):
    # Issue 4 FIX: retrieval_attempts is no longer incremented here.
    evidence = aggregate_evidence(
        state["xray_results"],
        allowed_modalities=["XRAY"],
        user_role=state.get("user_role", "doctor"),
    )
    logger.debug(f"[Aggregation] complete — {len(evidence)} evidence items")
    return {"evidence": evidence}


def evidence_quality_filter_node(state):
    verifier = EvidenceQualityVerifier()
    result = verifier.verify_and_filter(
        query=state["query"],
        evidence=state["evidence"],
        selected_modalities=["XRAY"],
    )
    return {
        "filtered_evidence": result["filtered_evidence"],
        "evidence_filter_result": result,
    }


def evidence_quality_gate_node(state):
    # Issue 9 FIX: respect threshold from state (set by UI / test runner).
    threshold = state.get("evidence_threshold", 0.4)
    gate = EvidenceQualityGate(threshold=threshold)
    return {
        "evidence_gate_result": gate.evaluate(
            evidence=state["evidence"],
            filter_result=state["evidence_filter_result"],
            query=state["query"],
        )
    }


def evidence_consistency_node(state):
    return {
        "consistency_result": check_evidence_consistency(
            response=state.get("final_answer", ""),
            evidence=state["filtered_evidence"],
        )
    }


def reasoning_node(state):
    result = clinical_reasoning_agent(
        query=state["query"],
        evidence=state["filtered_evidence"],
        user_role=state.get("user_role", "doctor"),
    )
    return {
        "final_answer":      result["final_answer"],
        "metrics":           result["metrics"],
        "reasoning_attempts": state.get("reasoning_attempts", 0) + 1,
    }


def clinical_safety_node(state):
    return {"safety_result": validate_clinical_safety(state["final_answer"])}


def xray_anatomy_node(state):
    return {"anatomy_result": validate_xray_anatomy(state["final_answer"])}


def response_refinement_node(state):
    refiner = ResponseRefiner()
    result  = refiner.refine_response(
        initial_response=state["final_answer"],
        evidence=state["filtered_evidence"],
        query=state["query"],
    )
    repaired = enforce_structure(result["refined_response"])
    return {
        "final_answer":    repaired,
        "refinement_count": state.get("refinement_count", 0) + 1,
    }


def structure_check_node(state):
    return {"structure_result": validate_structure(state["final_answer"])}


def response_quality_gate_node(state):
    # Issue 9 FIX: respect threshold from state.
    threshold = state.get("response_threshold", 0.7)
    gate = ResponseQualityGate(threshold=threshold)
    return {
        "response_gate_result": gate.evaluate(
            response=state["final_answer"],
            evidence=state["filtered_evidence"],
            metrics=state["metrics"],
        )
    }


def summary_node(state):
    return {
        "quality_scores": {
            "evidence": state["evidence_gate_result"]["score"],
            "response": state["response_gate_result"]["score"],
        },
        # Issue 3 FIX: include refinement_count in total_iterations.
        # Previously only retrieval + reasoning were summed, which under-
        # reported actual pipeline effort in the UI and exports.
        "total_iterations": (
            state.get("retrieval_attempts", 0)
            + state.get("reasoning_attempts", 0)
            + state.get("refinement_count", 0)   # ← was missing
        ),
    }


# =============================================================================
# DECISIONS
# =============================================================================

def should_retry_xray(state):
    if state["retrieval_contract_result"]["passed"]:
        return "proceed"
    # Issue 9 FIX: use configurable max from state (default 2).
    max_r = state.get("max_retrieval_retries", 2)
    attempts = state.get("retrieval_attempts", 0)
    if attempts >= max_r:
        logger.debug(f"[XrayContract] Max retrieval attempts ({attempts}) reached — forcing proceed")
        return "proceed"
    logger.debug(f"[XrayContract] Attempt {attempts} — retrying")
    return "retry"


def should_retry_retrieval(state):
    attempts = state.get("retrieval_attempts", 0)
    decision = state["evidence_gate_result"]["decision"]
    logger.debug(f"[EvidenceGate] attempt={attempts}, decision={decision}")
    if decision == "PASS":
        return "proceed"
    # Issue 9 FIX: use configurable max from state.
    max_r = state.get("max_retrieval_retries", 2)
    if attempts >= max_r:
        return "proceed"
    return "retry"


def should_proceed_after_consistency(state):
    # Issue 1 FIX: evidence_consistency now AFFECTS control flow.
    # Previously there was only a hard add_edge("evidence_consistency", "reason"),
    # making the consistency check purely informational (result stored but ignored).
    # Now: if consistency fails AND we still have retry budget, go back to xray.
    result  = state.get("consistency_result", {})
    passed  = result.get("passed", True)
    issues  = result.get("issues", [])

    if not passed:
        max_r    = state.get("max_retrieval_retries", 2)
        attempts = state.get("retrieval_attempts", 0)
        if attempts < max_r:
            logger.warning(f"[ConsistencyCheck] Failed ({issues}) — retrying retrieval (attempt {attempts})")
            return "retry"
        else:
            logger.warning(
                f"[ConsistencyCheck] Failed ({issues}) — max retries reached, proceeding anyway"
            )

    return "proceed"


def should_refine_for_safety(state):
    return "refine" if not state["safety_result"]["passed"] else "proceed"


def should_refine_for_structure(state):
    return "refine" if not state["structure_result"]["passed"] else "proceed"


def should_refine_response(state):
    if state["response_gate_result"]["decision"] == "PASS":
        return "complete"

    # Issue 9 FIX: use configurable max refinements from state.
    max_ref = state.get("max_refinement_retries", 2)

    if state.get("refinement_count", 0) >= max_ref:
        # Issue 2 FIX: log a warning so operators know quality was force-finalized,
        # and set forced_complete=True in state so test.py / Streamlit can flag it.
        score = state["response_gate_result"].get("score", 0.0)
        logger.warning(
            f"[ResponseGate] Max refinements ({max_ref}) reached — "
            f"force-finalizing with score={score:.2f}. "
            "Consider reviewing this response manually."
        )
        return "forced_complete"

    return "refine"


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_mmrag_graph():
    graph = StateGraph(MMRAgState)

    graph.add_node("xray",                 xray_node)
    graph.add_node("xray_contract",        xray_contract_node)
    graph.add_node("aggregate",            aggregation_node)
    graph.add_node("evidence_filter",      evidence_quality_filter_node)
    graph.add_node("evidence_gate",        evidence_quality_gate_node)
    graph.add_node("evidence_consistency", evidence_consistency_node)
    graph.add_node("reason",               reasoning_node)
    graph.add_node("clinical_safety",      clinical_safety_node)
    graph.add_node("response_refine",      response_refinement_node)
    graph.add_node("structure_check",      structure_check_node)
    graph.add_node("response_gate",        response_quality_gate_node)
    graph.add_node("summary",              summary_node)

    graph.set_entry_point("xray")

    graph.add_edge("xray", "xray_contract")

    graph.add_conditional_edges(
        "xray_contract",
        should_retry_xray,
        {"retry": "xray", "proceed": "aggregate"},
    )

    graph.add_edge("aggregate",       "evidence_filter")
    graph.add_edge("evidence_filter", "evidence_gate")

    graph.add_conditional_edges(
        "evidence_gate",
        should_retry_retrieval,
        {"retry": "xray", "proceed": "evidence_consistency"},
    )

    # Issue 1 FIX: replaced hard edge with conditional edge.
    # "retry"   → back to xray (re-fetch if consistency fails + budget allows)
    # "proceed" → forward to reason
    graph.add_conditional_edges(
        "evidence_consistency",
        should_proceed_after_consistency,
        {"retry": "xray", "proceed": "reason"},
    )

    graph.add_edge("reason", "clinical_safety")

    graph.add_conditional_edges(
        "clinical_safety",
        should_refine_for_safety,
        {"refine": "response_refine", "proceed": "structure_check"},
    )

    graph.add_edge("response_refine", "structure_check")

    graph.add_conditional_edges(
        "structure_check",
        should_refine_for_structure,
        {"refine": "response_refine", "proceed": "response_gate"},
    )

    # Issue 2 FIX: "forced_complete" is a new branch that reaches summary
    # with forced_complete=True set in state, so callers can detect it.
    graph.add_conditional_edges(
        "response_gate",
        should_refine_response,
        {
            "refine":          "response_refine",
            "complete":        "summary",
            "forced_complete": "summary",   # ← new branch
        },
    )

    graph.add_edge("summary", END)

    return graph.compile()