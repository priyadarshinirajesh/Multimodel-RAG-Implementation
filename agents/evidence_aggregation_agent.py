# agents/evidence_aggregation_agent.py

from utils.logger import get_logger
from agents.pathology_detection_agent import get_pathology_detector  # NEW!
from utils.report_parser import split_report_sections

logger = get_logger("EvidenceAggregator")


def aggregate_evidence(results, allowed_modalities=None, user_role="doctor"):
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

        sections = split_report_sections(payload.get("report_text", ""))

        print("======== Evidence Record =======")
        print("Indication: ", sections["indication"])
        print("Comparison: ", sections["comparison"])
        print("Findings: ", sections["findings"])
        print("Impression: ", sections["impression"])
        print("================================")

        evidence.append({
            "patient_id": payload.get("patient_id"),
            "modality": modality,
            "organ": payload.get("organ"),
            "report_text": payload.get("report_text", ""),
            "indication": payload.get("indication", sections["indication"]),
            "comparison": payload.get("comparison", sections["comparison"]),
            "findings": payload.get("findings", sections["findings"]),
            "impression": payload.get("impression", sections["impression"]),
            "image_path": payload.get("image_path"),
            "has_image": has_image
        })

        

    logger.info(
        f"[EvidenceAggregator] Final evidence count: {len(evidence)}"
    )

    # 🆕 NEW: Add pathology detection to evidence
    detector = get_pathology_detector()
    evidence = detector.analyze_evidence(evidence)
    
    logger.info(
        f"[EvidenceAggregator] Pathology detection completed"
    )

    # Apply RBAC filtering
    
    
    logger.info(
        f"[EvidenceAggregator] Final evidence count after RBAC: {len(evidence)}"
    )

    return evidence