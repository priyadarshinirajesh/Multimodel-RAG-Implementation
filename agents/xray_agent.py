# agents/xray_agent.py

from agents.retrieval_utils import retrieve_patient_records
from utils.logger import get_logger

logger = get_logger("XRAYAgent")

def xray_agent(patient_id: int, query: str):
    logger.info(f"Retrieving XRAY data for patient_id={patient_id}")
    results = retrieve_patient_records(
        patient_id=patient_id,
        query=query,
        modality="XRAY"
    )
    logger.info(f"Retrieved {len(results)} XRAY records")
    return results
