# vectorstore/qdrant_setup.py

from qdrant_client import QdrantClient, models
from pathlib import Path

# Persistent Qdrant storage
QDRANT_PATH = Path("data/qdrant")

client = QdrantClient(
    path=QDRANT_PATH
)

def create_collection():
    if not client.collection_exists("clinical_mmrag"):
        client.create_collection(
            collection_name="clinical_mmrag",
            vectors_config={
                "image": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                ),
                "text": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                ),
            },
        )
