# src/orchestrator.py
import pandas as pd
import os
from tools.retrieval import retrieve_text, retrieve_image
from rbac import apply_rbac

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "data", "raw", "final_multimodal_dataset.csv")

meta = pd.read_csv(META)

def orchestrate(query, role="doctor", patient_id=None, top_k_text=5, top_k_img=5):
    print("\n================ ORCHESTRATOR DEBUG ================")
    print(f"Incoming Query: {query}")
    print(f"Role: {role}")
    print(f"Patient ID: {patient_id}")
    print(f"CSV Loaded: {META}")

    # STEP 1 — Patient direct records
    if patient_id is not None:
        patient_records = meta[meta["patient_id"] == int(patient_id)]
        patient_records_list = patient_records.to_dict("records")
        print(f"-- Direct patient records found: {len(patient_records_list)} --")
    else:
        patient_records_list = []

    # STEP 2 — Global vector retrieval
    text_hits = retrieve_text(query, k=top_k_text)
    img_hits = retrieve_image(query, k=top_k_img)

    print("\n-- Text hits returned --")
    for h in text_hits: print(h)
    print("\n-- Image hits returned --")
    for h in img_hits: print(h)

    # STEP 3 — Merge patient-first retrieval
    retrieved = []

    for row in patient_records_list:
        retrieved.append({
            "filename": row["filename"],
            "patient_id": row["patient_id"],
            "modality": row["modality"],
            "findings": row.get("findings", ""),
            "impression": row.get("impression", ""),
            "source": "patient_record",
            "score": 1.0,
        })

    for t in text_hits + img_hits:
        if patient_id is not None and int(t["patient_id"]) != int(patient_id):
            continue

        retrieved.append({
            "filename": t["filename"],
            "patient_id": t["patient_id"],
            "modality": t.get("modality", ""),
            "findings": t.get("findings", ""),
            "impression": t.get("impression", ""),
            "score": t.get("score", 0),
            "source": "vector_hit",
        })

    print("\n-- Retrieved AFTER patient_id filtering --")
    for r in retrieved: print(r)

    # STEP 4 — RBAC
    filtered = apply_rbac(role, retrieved)
    print("\n-- Retrieved AFTER RBAC --")
    for r in filtered: print(r)
    print("=====================================================\n")

    return {
        "query": query,
        "role": role,
        "patient_id": patient_id,
        "retrieved": filtered
    }
