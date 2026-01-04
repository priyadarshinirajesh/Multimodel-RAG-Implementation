# agents/evidence_aggregation_agent.py

from utils.logger import get_logger

logger = get_logger("EvidenceAggregator")


def aggregate_evidence(results, allowed_modalities=None):
    evidence = []

    logger.info(
        f"[EvidenceAggregator] Aggregating {len(results)} raw retrieval results"
    )

    for r in results:

        # Handle (score, ScoredPoint)
        if isinstance(r, tuple):
            _, r = r

        # Qdrant ScoredPoint
        if hasattr(r, "payload"):
            payload = r.payload

        # Already normalized dict
        elif isinstance(r, dict):
            payload = r

        else:
            logger.warning(f"Skipping unknown type: {type(r)}")
            continue

        modality = payload.get("modality")

        # 🔒 HARD MODALITY ENFORCEMENT
        if allowed_modalities and modality not in allowed_modalities:
            logger.warning(
                f"Skipping evidence due to modality mismatch: {modality}"
            )
            continue

        has_image = payload.get("image_path") is not None

        logger.debug(
            f"[EvidenceAggregator] Record | modality={modality} | has_image={has_image}"
        )

        evidence.append({
            "patient_id": payload.get("patient_id"),
            "modality": modality,
            "organ": payload.get("organ"),
            "report_text": payload.get("report_text", ""),
            "image_path": payload.get("image_path"),
            "has_image": has_image
        })

    logger.info(
        f"[EvidenceAggregator] Final evidence count: {len(evidence)}"
    )

    return evidence
