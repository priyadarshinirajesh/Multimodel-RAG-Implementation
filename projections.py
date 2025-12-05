import pandas as pd
import os

CSV_PATH = "data/raw/dataset_indiana/indiana_projections.csv"
BASE_PATH = "C:/Users/priya/final_year_project/project_work/Multimodel-RAG-Implementation/data/raw/dataset_indiana/images/images_normalized"

df = pd.read_csv(CSV_PATH)

df["filename"] = df["filename"].apply(
    lambda f: os.path.join(BASE_PATH, f).replace("\\", "/")
)

df.to_csv(CSV_PATH, index=False)

print("✅ Paths updated successfully!")
