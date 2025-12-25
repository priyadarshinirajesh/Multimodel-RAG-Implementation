# agents/orchestrator.py

from agents.modality_router_agent import route_modalities
from agents.xray_agent import xray_agent
from agents.ct_agent import ct_agent
from agents.mri_agent import mri_agent
from agents.evidence_aggregation_agent import aggregate_evidence
from agents.clinical_reasoning_agent import clinical_reasoning_agent

def run_mmrag_pipeline(patient_id: int, query: str):
    modalities = route_modalities(query)
    all_results = []

    if "XRAY" in modalities:
        all_results.append(xray_agent(patient_id, query))
    if "CT" in modalities:
        all_results.append(ct_agent(patient_id, query))
    if "MRI" in modalities:
        all_results.append(mri_agent(patient_id, query))

    evidence = aggregate_evidence(all_results)

    final_response = clinical_reasoning_agent(query, evidence)

    return final_response, evidence

