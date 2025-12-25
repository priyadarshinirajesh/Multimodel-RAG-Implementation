# agents/modality_router_agent.py

from utils.logger import get_logger

logger = get_logger("RouterAgent")

def route_modalities(query: str):
    logger.info(f"Received query: {query}")

    q = query.lower()

    if "pancreas" in q:
        modalities = ["CT"]
    elif "prostate" in q:
        modalities = ["MRI"]
    elif "lung" in q or "chest" in q or "pulmonary" in q:
        modalities = ["XRAY"]
    else:
        modalities = ["XRAY", "CT", "MRI"]

    logger.info(f"Selected modalities: {modalities}")
    return modalities

