# agents/evidence_aggregation_agent.py

from utils.logger import get_logger
from agents.pathology_detection_agent import get_pathology_detector
from utils.report_parser import split_report_sections

logger = get_logger("EvidenceAggregator")


def aggregate_evidence(results, allowed_modalities=None, user_role="doctor"):
    evidence = []

    logger.info(
        f"[EvidenceAggregator] Aggregating {len(results)} raw retrieval results"
    )

    for r in results:
        # Handle (score, ScoredPoint) tuples
        if isinstance(r, tuple):
            _, r = r

        # Qdrant ScoredPoint
        if hasattr(r, "payload"):
            payload = r.payload

        # Already normalised dict
        elif isinstance(r, dict):
            payload = r

        else:
            logger.warning(f"Skipping unknown type: {type(r)}")
            continue

        modality = payload.get("modality")

        # Hard modality enforcement
        if allowed_modalities and modality not in allowed_modalities:
            logger.warning(
                f"[EvidenceAggregator] Skipping evidence — modality mismatch: {modality}"
            )
            continue

        has_image = payload.get("image_path") is not None

        logger.debug(
            f"[EvidenceAggregator] Record | modality={modality} | has_image={has_image}"
        )

        sections = split_report_sections(payload.get("report_text", ""))

        # Issue 8 FIX: replaced per-record print() block with a single logger.debug()
        # call.  The previous 6-line print block ran for every evidence record on
        # every pipeline invocation, flooding stdout during batch runs (e.g. 100
        # queries × 7 records = 700 print blocks) and hiding real log messages.
        # logger.debug() only emits output when DEBUG-level logging is enabled.
        logger.debug(
            f"[EvidenceAggregator] sections — "
            f"indication='{sections['indication'][:80]}' | "
            f"findings='{sections['findings'][:80]}' | "
            f"impression='{sections['impression'][:80]}'"
        )

        evidence.append({
            "patient_id":  payload.get("patient_id"),
            "modality":    modality,
            "organ":       payload.get("organ"),
            "report_text": payload.get("report_text", ""),
            "indication":  payload.get("indication",  sections["indication"]),
            "comparison":  payload.get("comparison",  sections["comparison"]),
            "findings":    payload.get("findings",    sections["findings"]),
            "impression":  payload.get("impression",  sections["impression"]),
            "image_path":  payload.get("image_path"),
            "has_image":   has_image,
        })

    logger.info(
        f"[EvidenceAggregator] Evidence count after modality filter: {len(evidence)}"
    )

    # Pathology detection (DenseNet-121)
    detector = get_pathology_detector()
    evidence = detector.analyze_evidence(evidence)

    logger.info("[EvidenceAggregator] Pathology detection completed")

    logger.info(
        f"[EvidenceAggregator] Final evidence count: {len(evidence)}"
    )

    return evidence