# tests/test_qdrant_basic.py

from vectorstore.qdrant_setup import client

# Check collection exists
collection = client.get_collection("clinical_mmrag")
print("Collection exists:", collection is not None)

# Count records
count = client.count("clinical_mmrag")
print("Total records in Qdrant:", count)
