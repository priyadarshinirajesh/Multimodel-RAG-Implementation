from utils.logger import get_logger
logger = get_logger("EvidenceAggregator")

def aggregate_evidence(results):
    """
    Normalize retrieved results into a clean evidence list.
    Supports:
    - Qdrant ScoredPoint
    - (score, ScoredPoint)
    - already-normalized dict
    """

    evidence = []

    logger.info(
        f"[EvidenceAggregator] Aggregating {len(results)} raw retrieval results"
    )

    for r in results:

        # Case 1: (score, ScoredPoint)
        if isinstance(r, tuple):
            _, r = r

        # Case 2: Qdrant ScoredPoint
        if hasattr(r, "payload"):
            payload = r.payload

        # Case 3: Already a dict (post-filtering)
        elif isinstance(r, dict):
            payload = r

        else:
            logger.warning(f"[EvidenceAggregator] Skipping unknown type: {type(r)}")
            continue

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
            "has_image": has_image,
            "score": payload.get("score")
        })

    logger.info(
        f"[EvidenceAggregator] Final evidence count: {len(evidence)}"
    )

    return evidence
