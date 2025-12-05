# combined_dataset.py

import pandas as pd
import os
import random

# ------------------------------------------------
# PATHS
# ------------------------------------------------
RAW_DIR = "data/raw"

INDIANA_DIR = os.path.join(RAW_DIR, "dataset_indiana")
CT_MRI_DIR = os.path.join(RAW_DIR, "CT-MRI")

# Input files
IND_REPORTS = os.path.join(INDIANA_DIR, "indiana_reports.csv")
IND_PROJ = os.path.join(INDIANA_DIR, "indiana_projections.csv")

CTMRI_REPORTS = os.path.join(CT_MRI_DIR, "ct_mri_reports.csv")
CTMRI_PROJ = os.path.join(CT_MRI_DIR, "ct_mri_projections.csv")

# Output
FINAL_OUT = os.path.join(RAW_DIR, "final_multimodal_dataset.csv")

# ------------------------------------------------
# 1. LOAD CHEST X-RAY (Indiana)
# ------------------------------------------------
cxr_reports = pd.read_csv(IND_REPORTS)
cxr_proj = pd.read_csv(IND_PROJ)

# Merge by uid
cxr = cxr_proj.merge(cxr_reports, on="uid", how="inner")

# Chest X-ray tags
cxr["modality"] = "X-Ray"
cxr["organ"] = "Chest"

# chest dataset's uid is treated as patient_id
cxr["patient_id"] = cxr["uid"]

# ------------------------------------------------
# 2. LOAD CT + MRI DATASETS
# ------------------------------------------------
ctmri_reports = pd.read_csv(CTMRI_REPORTS)
ctmri_proj = pd.read_csv(CTMRI_PROJ)

ctmri = ctmri_proj.merge(ctmri_reports, on="uid", how="left")

# Identify modality from path
ctmri["modality"] = ctmri["filename"].apply(
    lambda x: "CT" if "pancreas" in x.lower() else "MRI"
)

# Identify organ
ctmri["organ"] = ctmri["filename"].apply(
    lambda x: "Pancreas" if "pancreas" in x.lower() else "Prostate"
)

# ------------------------------------------------
# 3. ASSIGN CT/MRI PATIENT IDs (STANDARD FOR MULTIMODAL RAG)
# ------------------------------------------------
# Get all indiana patient IDs
indiana_patients = list(cxr["patient_id"].unique())

# Random assignment of all CT/MRI slices
ctmri["patient_id"] = [
    random.choice(indiana_patients) for _ in range(len(ctmri))
]

# ------------------------------------------------
# 4. COMBINE EVERYTHING
# ------------------------------------------------
combined = pd.concat([cxr, ctmri], ignore_index=True)

# Final column order
combined = combined[
    [
        "patient_id", "uid", "modality", "organ",
        "filename", "projection",
        "MeSH", "Problems", "indication",
        "comparison", "findings", "impression"
    ]
]

# ------------------------------------------------
# 5. Save final dataset
# ------------------------------------------------
combined.to_csv(FINAL_OUT, index=False)

print("\n🔥 FINAL MULTIMODAL DATASET CREATED!")
print(f"📁 Saved at: {FINAL_OUT}")
print(f"🧪 Total rows: {len(combined)}")
print(f"👥 Patients: {len(indiana_patients)}")
print("Done.\n")