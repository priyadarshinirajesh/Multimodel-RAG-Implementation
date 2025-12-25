# ingestion/ingest_to_qdrant.py

from qdrant_client.models import PointStruct
from ingestion.preprocess_dataset import load_and_preprocess
from embeddings.image_embeddings import embed_image
from embeddings.text_embeddings import embed_text
from vectorstore.qdrant_setup import client, create_collection

def ingest():
    df = load_and_preprocess()
    create_collection()

    points = []

    for idx, row in df.iterrows():
        # 🔹 IMAGE EMBEDDING
        image_vector = embed_image(row["filename"])

        # 🔹 TEXT EMBEDDING
        text_vector = embed_text(row["report_text"])

        payload = {
            "patient_id": int(row["patient_id"]),
            "uid": int(row["uid"]),
            "modality": row["modality"],
            "organ": row["organ"],
            "projection": row["projection"],
            "MeSH": row["MeSH"],
            "Problems": row["Problems"],
            "role": row["role"],
            "image_path": row["filename"],
            "report_text": row["report_text"],
        }

        points.append(
            PointStruct(
                id=idx,
                vector={
                    "image": image_vector,
                    "text": text_vector,
                },
                payload=payload,
            )
        )

    client.upload_points(
        collection_name="clinical_mmrag",
        points=points,
    )

    print(f"✅ Ingested {len(points)} multimodal records")

if __name__ == "__main__":
    ingest()

