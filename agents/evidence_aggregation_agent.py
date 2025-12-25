# agents/evidence_aggregation_agent.py

from utils.logger import get_logger
logger = get_logger("EvidenceAggregator")

def aggregate_evidence(results):
    evidence = []

    logger.info(
        f"[EvidenceAggregator] Aggregating {len(results)} raw retrieval results"
    )

    for r in results:
        # Handle Qdrant returning (score, point)
        if isinstance(r, tuple):
            _, point = r
        else:
            point = r

        payload = point.payload

        has_image = payload.get("image_path") is not None

        logger.debug(
            f"[EvidenceAggregator] Record | modality={payload.get('modality')} | "
            f"has_image={has_image}"
        )

        evidence.append({
            "patient_id": payload.get("patient_id"),
            "modality": payload.get("modality"),
            "organ": payload.get("organ"),
            "report_text": payload.get("report_text", ""),
            "image_path": payload.get("image_path"),
            "has_image": has_image
        })

    logger.info(
        f"[EvidenceAggregator] Final evidence count: {len(evidence)}"
    )

    return evidence
