# ingestion/preprocess_dataset.py

import pandas as pd

CSV_PATH = "data/raw/final_multimodal_dataset.csv"

def load_and_preprocess():
    df = pd.read_csv(CSV_PATH)

    # Fill NaNs safely
    df = df.fillna("")

    # Build ONE canonical clinical text field
    df["report_text"] = (
        "Indication: " + df["indication"] + "\n"
        "Comparison: " + df["comparison"] + "\n"
        "Findings: " + df["findings"] + "\n"
        "Impression: " + df["impression"]
    )

    # Normalize modality (X-Ray → XRAY)
    df["modality"] = (
        df["modality"]
        .str.upper()
        .str.replace("-", "")
        .str.replace(" ", "")
    )

    # Add role (static for now)
    df["role"] = "doctor"

    return df

