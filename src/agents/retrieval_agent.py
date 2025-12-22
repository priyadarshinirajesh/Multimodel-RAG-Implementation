# src/agents/retrieval_agent.py

import os
import sys
import pandas as pd

# Make sure src is visible
sys.path.append(os.path.abspath("src"))

from tools.retrieval import retrieve_text, retrieve_image
from rbac import apply_rbac

# Correct root (go two levels up: src/agents → src → project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(PROJECT_ROOT, "data", "raw", "final_multimodal_dataset.csv")

meta = pd.read_csv(META)


class RetrievalAgent:
    def run(self, query, patient_id, role, k_text=5, k_img=5):
        print("\n[RetrievalAgent] Running retrieval...")

        # Direct patient rows
        patient_records = meta[meta["patient_id"] == int(patient_id)]
        records_list = patient_records.to_dict("records")

        # Vector search
        text_hits = retrieve_text(query, k=k_text)
        img_hits = retrieve_image(query, k=k_img)

        # Filter: only same patient
        filtered_hits = [
            h for h in text_hits + img_hits
            if int(h["patient_id"]) == int(patient_id)
        ]

        # Build evidence
        evidence = []

        # Add real patient rows
        for row in records_list:
            evidence.append({
                "filename": row["filename"],
                "modality": row["modality"],
                "findings": row.get("findings", ""),
                "impression": row.get("impression", ""),
                "source": "patient_record",
                "score": 1.0
            })

        # Add retrieved rows
        for h in filtered_hits:
            evidence.append({
                "filename": h["filename"],
                "modality": h.get("modality", ""),
                "findings": h.get("findings", ""),
                "impression": h.get("impression", ""),
                "source": "retriever_hit",
                "score": h.get("score", 0)
            })

        # RBAC
        evidence = apply_rbac(role, evidence)

        print(f"[RetrievalAgent] Evidence returned: {len(evidence)} items\n")
        return evidence