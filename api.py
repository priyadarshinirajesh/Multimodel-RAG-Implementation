# api.py
# Place this in your project root (same level as streamlit_app.py)
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agents.langgraph_flow.mmrag_graph import build_mmrag_graph

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = FastAPI(
    title="MedAI Clinical Decision Support API",
    description="Multimodal RAG system for clinical decision support",
    version="1.0.0",
)

# Allow the HTML file to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML frontend at /
# Place your medical_ai.html in a folder called "frontend/"
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)

# Build the graph ONCE at startup (expensive — do not rebuild per request)
print("⏳ Building MM-RAG graph...")
graph = build_mmrag_graph()
print("✅ Graph ready.")


# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    patient_id: str                        # e.g. "P-1001" or "1001"
    query: str
    user_role: str = "doctor"
    evidence_threshold: float = 0.4
    response_threshold: float = 0.7
    max_retrieval_retries: int = 2
    max_refinement_retries: int = 2


# ──────────────────────────────────────────────
# HELPER — make numpy types JSON-serialisable
# ──────────────────────────────────────────────

def _serialise(obj):
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, dict):         return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [_serialise(i) for i in obj]
    return obj


def _parse_patient_id(raw: str) -> int:
    """Accept 'P-1001', '1001', or 1001 — return int."""
    cleaned = str(raw).upper().replace("P-", "").replace("P", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid patient_id: '{raw}'. Expected format: P-1001 or 1001."
        )


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {"message": "MedAI API is running. Place your index.html in the frontend/ folder."},
        status_code=200,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "graph": "ready"}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint — mirrors what streamlit_app.py does.
    Accepts the same fields as the HTML frontend sends.
    Returns a JSON object the frontend already knows how to render.
    """

    patient_id_int = _parse_patient_id(req.patient_id)

    initial_state = {
        "patient_id":   patient_id_int,
        "query":        req.query,
        "user_role":    req.user_role,

        # UI-configurable thresholds (Issue 9)
        "evidence_threshold":     req.evidence_threshold,
        "response_threshold":     req.response_threshold,
        "max_retrieval_retries":  req.max_retrieval_retries,
        "max_refinement_retries": req.max_refinement_retries,

        # Routing
        "modalities":           ["XRAY"],
        "routing_verification": {},
        "routing_gate_result":  {},

        # Retrieval
        "xray_results": [],
        "ct_results":   [],
        "mri_results":  [],

        # Evidence
        "evidence":               [],
        "filtered_evidence":      [],
        "evidence_filter_result": {},
        "evidence_gate_result":   {},
        "retrieval_attempts":     0,

        # Reasoning
        "final_answer":       "",
        "metrics":            {},
        "response_gate_result": {},
        "refinement_result":    {},
        "reasoning_attempts":   0,
        "refinement_count":     0,
        "forced_complete":      False,

        # Global
        "total_iterations": 0,
        "quality_scores":   {},
    }

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # ── Build the response the HTML frontend expects ──────────────────────────

    metrics        = final_state.get("metrics", {}) or {}
    quality_scores = final_state.get("quality_scores", {}) or {}
    filtered_ev    = final_state.get("filtered_evidence", []) or []

    # Strip keys the frontend should not see
    hidden = {"GroundednessSimple", "EvaluationNote"}
    clean_metrics = {k: v for k, v in metrics.items() if k not in hidden}

    # Serialise evidence (remove non-JSON-safe types, keep what the UI needs)
    evidence_out = []
    for e in filtered_ev:
        evidence_out.append({
            "modality":         e.get("modality", "N/A"),
            "organ":            e.get("organ", ""),
            "report_text":      e.get("report_text", ""),
            "relevance_score":  float(e.get("relevance_score", 0.0)),
            "has_image":        bool(e.get("has_image", False)),
            "image_path":       e.get("image_path"),
            "pathology_scores": _serialise(e.get("pathology_scores", {})),
            "top_pathologies":  _serialise(e.get("top_pathologies", [])),
            "pathology_findings": e.get("pathology_findings", ""),
        })

    response = {
        # Pipeline counters
        "total_iterations":   int(final_state.get("total_iterations", 0)),
        "retrieval_attempts": int(final_state.get("retrieval_attempts", 0)),
        "reasoning_attempts": int(final_state.get("reasoning_attempts", 0)),
        "refinement_count":   int(final_state.get("refinement_count", 0)),
        "forced_complete":    bool(final_state.get("forced_complete", False)),

        # Quality
        "quality_scores": _serialise(quality_scores),
        "evidence_gate_result": _serialise(final_state.get("evidence_gate_result", {})),
        "response_gate_result": _serialise(final_state.get("response_gate_result", {})),
        "evidence_filter_result": {
            "quality_score":  float((final_state.get("evidence_filter_result") or {}).get("quality_score", 0)),
            "removed_count":  int((final_state.get("evidence_filter_result") or {}).get("removed_count", 0)),
            "feedback":       (final_state.get("evidence_filter_result") or {}).get("feedback", ""),
        },
        "consistency_result": _serialise(final_state.get("consistency_result", {})),

        # Clinical answer
        "final_answer": final_state.get("final_answer", ""),

        # Metrics
        "metrics": _serialise(clean_metrics),

        # Evidence
        "evidence":          [{"modality": e.get("modality")} for e in filtered_ev],
        "filtered_evidence": evidence_out,
    }

    return JSONResponse(content=response)


@app.get("/api/image")
async def serve_image(path: str):
    """
    Serve an X-ray image by its file path.
    The frontend calls: /api/image?path=/absolute/path/to/image.png
    """
    img_path = Path(path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Basic security: only serve files inside your data directory
    data_root = Path(__file__).parent / "data"
    try:
        img_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(img_path)