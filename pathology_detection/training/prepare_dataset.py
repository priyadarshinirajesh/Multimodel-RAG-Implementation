# pathology_detection/training/prepare_dataset.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# Define 14 common thoracic pathologies
PATHOLOGY_CLASSES = [
    'Atelectasis',
    'Cardiomegaly', 
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

def extract_labels_from_mesh(mesh_str):
    """
    Extract binary labels from MeSH string
    
    Example MeSH: "Cardiomegaly/borderline;Pulmonary Artery/enlarged"
    Returns: {'Cardiomegaly': 1, 'Atelectasis': 0, ...}
    """
    if pd.isna(mesh_str):
        return {pathology: 0 for pathology in PATHOLOGY_CLASSES}
    
    mesh_lower = str(mesh_str).lower()
    labels = {}
    
    for pathology in PATHOLOGY_CLASSES:
        # Check if pathology is mentioned in MeSH or Problems
        if pathology.lower() in mesh_lower:
            labels[pathology] = 1
        else:
            labels[pathology] = 0
    
    return labels

def extract_labels_from_problems(problems_str):
    """Extract labels from Problems column"""
    if pd.isna(problems_str):
        return {pathology: 0 for pathology in PATHOLOGY_CLASSES}
    
    problems_lower = str(problems_str).lower()
    labels = {}
    
    for pathology in PATHOLOGY_CLASSES:
        if pathology.lower() in problems_lower:
            labels[pathology] = 1
        else:
            labels[pathology] = 0
    
    return labels

def combine_labels(mesh_labels, problem_labels):
    """Combine labels from both sources (logical OR)"""
    combined = {}
    for pathology in PATHOLOGY_CLASSES:
        combined[pathology] = max(
            mesh_labels.get(pathology, 0),
            problem_labels.get(pathology, 0)
        )
    return combined

def prepare_dataset(csv_path, output_dir, test_size=0.15, val_size=0.15):
    """
    Main function to prepare labeled dataset
    
    Args:
        csv_path: Path to final_multimodal_dataset.csv
        output_dir: Where to save train/val/test splits
        test_size: Fraction for test set
        val_size: Fraction for validation set
    """
    
    print("=" * 80)
    print("DATASET PREPARATION FOR PATHOLOGY DETECTION")
    print("=" * 80)
    
    # 1. Load dataset
    print(f"\n[1/6] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total records: {len(df)}")
    print(f"   Unique patients: {df['patient_id'].nunique()}")
    
    # 2. Filter valid images only
    print("\n[2/6] Filtering valid image records...")
    df = df[df['filename'].notna()].copy()
    print(f"   Records with images: {len(df)}")
    
    # 3. Extract labels from MeSH and Problems columns
    print("\n[3/6] Extracting pathology labels...")
    
    mesh_labels = df['MeSH'].apply(extract_labels_from_mesh)
    problem_labels = df['Problems'].apply(extract_labels_from_problems)
    
    # Combine labels (if either MeSH or Problems has the label, mark as positive)
    combined_labels = [
        combine_labels(m, p) 
        for m, p in zip(mesh_labels, problem_labels)
    ]
    
    # Convert to DataFrame
    label_df = pd.DataFrame(combined_labels)
    
    # Add to main dataframe
    for pathology in PATHOLOGY_CLASSES:
        df[f'label_{pathology}'] = label_df[pathology]
    
    # 4. Analyze class distribution
    print("\n[4/6] Class distribution:")
    print("-" * 60)
    
    class_stats = []
    for pathology in PATHOLOGY_CLASSES:
        count = df[f'label_{pathology}'].sum()
        percentage = (count / len(df)) * 100
        class_stats.append({
            'Pathology': pathology,
            'Count': count,
            'Percentage': f'{percentage:.2f}%'
        })
        print(f"   {pathology:20s}: {count:5d} ({percentage:5.2f}%)")
    
    # Save class distribution
    stats_df = pd.DataFrame(class_stats)
    stats_df.to_csv(f"{output_dir}/class_distribution.csv", index=False)
    
    # 5. Create stratified splits (patient-level to avoid data leakage)
    print("\n[5/6] Creating train/val/test splits...")
    
    # Group by patient to avoid same patient in different splits
    patient_groups = df.groupby('patient_id').agg({
        'filename': 'first',  # Take first image per patient for splitting
        **{f'label_{p}': 'max' for p in PATHOLOGY_CLASSES}  # Max label per patient
    }).reset_index()
    
    # Calculate multi-label for stratification
    # Use most common pathology as stratification key
    patient_groups['stratify_key'] = patient_groups[[f'label_{p}' for p in PATHOLOGY_CLASSES]].sum(axis=1)
    patient_groups['stratify_key'] = patient_groups['stratify_key'].apply(
        lambda x: min(x, 3)  # Cap at 3 to avoid too many strata
    )
    
    # Split patients first (to avoid data leakage)
    train_patients, test_patients = train_test_split(
        patient_groups['patient_id'],
        test_size=test_size,
        random_state=42,
        stratify=patient_groups['stratify_key']
    )
    
    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=val_size / (1 - test_size),
        random_state=42
    )
    
    # Create splits based on patient IDs
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"   Train set: {len(train_df)} images from {len(train_patients)} patients")
    print(f"   Val set:   {len(val_df)} images from {len(val_patients)} patients")
    print(f"   Test set:  {len(test_df)} images from {len(test_patients)} patients")
    
    # 6. Save processed datasets
    print("\n[6/6] Saving processed datasets...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select relevant columns
    columns_to_save = [
        'patient_id', 'uid', 'filename', 'projection',
        'MeSH', 'Problems', 'findings', 'impression'
    ] + [f'label_{p}' for p in PATHOLOGY_CLASSES]
    
    train_df[columns_to_save].to_csv(f"{output_dir}/train_labels.csv", index=False)
    val_df[columns_to_save].to_csv(f"{output_dir}/val_labels.csv", index=False)
    test_df[columns_to_save].to_csv(f"{output_dir}/test_labels.csv", index=False)
    
    # Save metadata
    metadata = {
        'pathology_classes': PATHOLOGY_CLASSES,
        'num_classes': len(PATHOLOGY_CLASSES),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_patients': int(len(train_patients)),
        'val_patients': int(len(val_patients)),
        'test_patients': int(len(test_patients))
    }
    
    with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Files saved to: {output_dir}")
    print("=" * 80)
    
    return train_df, val_df, test_df, metadata


if __name__ == "__main__":
    # Paths
    CSV_PATH = "data/raw/final_multimodal_dataset.csv"
    OUTPUT_DIR = "data/processed"
    
    # Run preparation
    train_df, val_df, test_df, metadata = prepare_dataset(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        test_size=0.15,
        val_size=0.15
    )
    
    print("\n📊 Dataset Summary:")
    print(f"   Training:   {len(train_df):5d} images")
    print(f"   Validation: {len(val_df):5d} images")
    print(f"   Testing:    {len(test_df):5d} images")
    print(f"   Total:      {len(train_df) + len(val_df) + len(test_df):5d} images")

