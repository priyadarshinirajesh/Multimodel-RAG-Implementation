# tests/test_patient_retrieval.py

from agents.retrieval_utils import retrieve_patient_records

results = retrieve_patient_records(
    patient_id=1,
    query="pulmonary abnormality",
    modality="XRAY"
)

for r in results:
    print("Patient ID:", r.payload["patient_id"])
    print("Modality:", r.payload["modality"])
    print("Text snippet:", r.payload["report_text"][:200])
    print("-" * 60)

