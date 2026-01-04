# agents/retrieval_utils.py

from qdrant_client import models
from vectorstore.qdrant_setup import client
from embeddings.text_embeddings import embed_text
from utils.logger import get_logger

logger = get_logger("RetrievalUtils")


def _filter_by_modality(points, modality):
    if not modality:
        return points
    return [
        p for p in points
        if hasattr(p, "payload")
        and p.payload
        and p.payload.get("modality") == modality
    ]


def retrieve_patient_records(
    patient_id: int,
    query: str,
    modality: str | None = None,
    limit: int = 5,
    include_image: bool = True
):
    logger.debug(f"Embedding query: {query}")
    query_vector = embed_text(query)

    must_conditions = [
        models.FieldCondition(
            key="patient_id",
            match=models.MatchValue(value=patient_id)
        )
    ]

    if modality:
        must_conditions.append(
            models.FieldCondition(
                key="modality",
                match=models.MatchValue(value=modality)
            )
        )

    query_filter = models.Filter(must=must_conditions)

    # -------- TEXT RETRIEVAL --------
    logger.info(f"[TextRetrieval] Searching TEXT vectors | modality={modality}")

    text_response = client.query_points(
        collection_name="clinical_mmrag",
        query=query_vector,
        using="text",
        query_filter=query_filter,
        limit=limit
    )

    text_points = (
        text_response.points
        if hasattr(text_response, "points")
        else text_response
    )

    text_points = _filter_by_modality(text_points, modality)

    # -------- IMAGE RETRIEVAL --------
    image_points = []

    if include_image:
        logger.info(f"[ImageRetrieval] Searching IMAGE vectors | modality={modality}")

        image_response = client.query_points(
            collection_name="clinical_mmrag",
            query=query_vector,
            using="image",
            query_filter=query_filter,
            limit=limit
        )

        image_points = (
            image_response.points
            if hasattr(image_response, "points")
            else image_response
        )

        image_points = _filter_by_modality(image_points, modality)

        for p in image_points:
            logger.debug(f"[ImageEvidence] {p.payload.get('image_path')}")

    # -------- MERGE UNIQUE --------
    all_points = {p.id: p for p in text_points}
    for p in image_points:
        all_points[p.id] = p

    logger.info(
        f"[RetrievalUtils] Final evidence count (after modality filter): {len(all_points)}"
    )

    return list(all_points.values())
