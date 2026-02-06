# pathology_detection/training/prepare_dataset.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import re
from collections import Counter

# ✅ Define anatomical terms to EXCLUDE (not pathologies)
ANATOMICAL_EXCLUSIONS = [
    'lung', 'right', 'left', 'bilateral', 'thoracic vertebrae', 'aorta', 
    'base', 'spine', 'diaphragm', 'thoracic', 'hilum', 'mediastinum',
    'apex', 'ribs', 'cardiac shadow', 'thorax', 'abdomen', 'heart',
    'trachea', 'pleura', 'costophrenic angle', 'lymph nodes', 'bone',
    'upper lobe', 'lower lobe', 'middle lobe', 'lingula', 'posterior',
    'anterior', 'lumbar vertebrae', 'bronchovascular', 'pulmonary',
    'catheters', 'indwelling', 'medical device', 'implanted medical device',
    'surgical instruments', 'technical quality of image unsatisfactory',
    'no indexing', 'large', 'small', 'multiple', 'scattered', 'diffuse',
    'focal', 'patchy', 'streaky', 'prominent', 'elevated', 'blunted',
    'flattened', 'tortuous', 'round', 'healed', 'chronic', 'acute',
    'degenerative', 'markings', 'density', 'opacity'  # These are too generic
]

# ✅ Define common pathology keywords to KEEP
PATHOLOGY_KEYWORDS = [
    'cardiomegaly', 'atelectasis', 'effusion', 'pneumonia', 'edema',
    'emphysema', 'consolidation', 'infiltrate', 'nodule', 'mass',
    'granuloma', 'fibrosis', 'pneumothorax', 'hernia', 'fracture',
    'scoliosis', 'spondylosis', 'atherosclerosis', 'calcinosis',
    'thickening', 'disease', 'congestion', 'hyperdistention',
    'hypoinflation', 'cicatrix', 'deformity', 'sclerosis', 'osteophyte'
]


def extract_pathologies_from_text(text):
    """
    Extract individual pathology terms from MeSH/Problems text
    """
    if pd.isna(text):
        return []
    
    text = str(text)
    
    # Split by common delimiters
    terms = re.split(r'[;/,\n]', text)
    
    pathologies = []
    for term in terms:
        term = term.strip()
        
        # Remove qualifiers
        qualifiers = ['borderline', 'mild', 'moderate', 'severe', 'enlarged', 'small']
        for q in qualifiers:
            term = term.replace(q, '').strip()
        
        # Skip empty or very short terms
        if len(term) < 3:
            continue
        
        # Skip common non-pathology terms
        skip_terms = ['normal', 'stable', 'unchanged', 'negative', 'clear', 'within normal limits']
        if any(skip in term.lower() for skip in skip_terms):
            continue
        
        pathologies.append(term.title())
    
    return pathologies


def is_valid_pathology(pathology_name):
    """
    Determine if a term is actually a pathology (not anatomy)
    """
    pathology_lower = pathology_name.lower()
    
    # ❌ Exclude anatomical structures
    if pathology_lower in ANATOMICAL_EXCLUSIONS:
        return False
    
    # ✅ Keep if it contains pathology keywords
    if any(keyword in pathology_lower for keyword in PATHOLOGY_KEYWORDS):
        return True
    
    # ❌ Exclude if it's purely anatomical
    return False


def discover_pathologies(df, min_occurrences=20):
    """
    Automatically discover ACTUAL pathologies from the dataset
    """
    
    all_pathologies = []
    
    # Extract from MeSH column
    if 'MeSH' in df.columns:
        for mesh in df['MeSH'].dropna():
            all_pathologies.extend(extract_pathologies_from_text(mesh))
    
    # Extract from Problems column
    if 'Problems' in df.columns:
        for problem in df['Problems'].dropna():
            all_pathologies.extend(extract_pathologies_from_text(problem))
    
    # Count occurrences
    pathology_counts = Counter(all_pathologies)
    
    # ✅ Filter by minimum occurrences AND validity
    valid_pathologies = [
        pathology for pathology, count in pathology_counts.items()
        if count >= min_occurrences and is_valid_pathology(pathology)
    ]
    
    # Sort by frequency
    valid_pathologies.sort(key=lambda p: pathology_counts[p], reverse=True)
    
    return valid_pathologies, pathology_counts


def extract_labels_from_text(text, pathology_classes):
    """Extract binary labels for all pathologies from text"""
    if pd.isna(text):
        return {pathology: 0 for pathology in pathology_classes}
    
    text_lower = str(text).lower()
    labels = {}
    
    for pathology in pathology_classes:
        if pathology.lower() in text_lower:
            labels[pathology] = 1
        else:
            labels[pathology] = 0
    
    return labels


def combine_labels(mesh_labels, problem_labels):
    """Combine labels from both sources (logical OR)"""
    combined = {}
    for pathology in mesh_labels.keys():
        combined[pathology] = max(
            mesh_labels.get(pathology, 0),
            problem_labels.get(pathology, 0)
        )
    return combined


def prepare_dataset(csv_path, output_dir, min_occurrences=20, test_size=0.15, val_size=0.15):
    """
    Main function to prepare labeled dataset with AUTOMATIC pathology discovery
    """
    
    print("=" * 80)
    print("AUTOMATIC PATHOLOGY DETECTION DATASET PREPARATION (FILTERED)")
    print("=" * 80)
    
    # 1. Load dataset
    print(f"\n[1/7] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total records: {len(df)}")
    print(f"   Unique patients: {df['patient_id'].nunique()}")
    
    # 2. Filter valid images only
    print("\n[2/7] Filtering valid image records...")
    df = df[df['filename'].notna()].copy()
    print(f"   Records with images: {len(df)}")
    
    # 3. DISCOVER PATHOLOGIES (with filtering)
    print(f"\n[3/7] Discovering ACTUAL pathologies (min_occurrences={min_occurrences})...")
    pathology_classes, pathology_counts = discover_pathologies(df, min_occurrences)
    
    print(f"\n   ✅ Discovered {len(pathology_classes)} valid pathologies:")
    print("-" * 60)
    for i, pathology in enumerate(pathology_classes, 1):
        count = pathology_counts[pathology]
        percentage = (count / len(df)) * 100
        print(f"   {i:2d}. {pathology:35s}: {count:5d} ({percentage:5.2f}%)")
    
    # 4. Extract labels
    print(f"\n[4/7] Extracting pathology labels...")
    
    mesh_labels = df['MeSH'].apply(lambda x: extract_labels_from_text(x, pathology_classes))
    problem_labels = df['Problems'].apply(lambda x: extract_labels_from_text(x, pathology_classes))
    
    combined_labels = [
        combine_labels(m, p) 
        for m, p in zip(mesh_labels, problem_labels)
    ]
    
    # Convert to DataFrame and add to main df
    label_df = pd.DataFrame(combined_labels)
    
    # ✅ FIX: Use pd.concat to avoid fragmentation warning
    label_columns = {f'label_{pathology}': label_df[pathology] for pathology in pathology_classes}
    df = pd.concat([df, pd.DataFrame(label_columns)], axis=1)
    
    # 5. Analyze final class distribution
    print("\n[5/7] Final class distribution:")
    print("-" * 60)
    
    class_stats = []
    for pathology in pathology_classes:
        count = df[f'label_{pathology}'].sum()
        percentage = (count / len(df)) * 100
        class_stats.append({
            'Pathology': pathology,
            'Count': int(count),
            'Percentage': f'{percentage:.2f}%'
        })
        print(f"   {pathology:35s}: {count:5d} ({percentage:5.2f}%)")
    
    # Save class distribution
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats_df = pd.DataFrame(class_stats)
    stats_df.to_csv(output_path / "class_distribution.csv", index=False)
    
    # 6. Create stratified splits
    print("\n[6/7] Creating train/val/test splits...")
    
    patient_groups = df.groupby('patient_id').agg({
        'filename': 'first',
        **{f'label_{p}': 'max' for p in pathology_classes}
    }).reset_index()
    
    patient_groups['stratify_key'] = patient_groups[[f'label_{p}' for p in pathology_classes]].sum(axis=1)
    patient_groups['stratify_key'] = patient_groups['stratify_key'].apply(lambda x: min(x, 5))
    
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
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"   Train set: {len(train_df)} images from {len(train_patients)} patients")
    print(f"   Val set:   {len(val_df)} images from {len(val_patients)} patients")
    print(f"   Test set:  {len(test_df)} images from {len(test_patients)} patients")
    
    # 7. Save processed datasets
    print("\n[7/7] Saving processed datasets...")
    
    columns_to_save = [
        'patient_id', 'uid', 'filename', 'projection',
        'MeSH', 'Problems', 'findings', 'impression'
    ] + [f'label_{p}' for p in pathology_classes]
    
    train_df[columns_to_save].to_csv(output_path / "train_labels.csv", index=False)
    val_df[columns_to_save].to_csv(output_path / "val_labels.csv", index=False)
    test_df[columns_to_save].to_csv(output_path / "test_labels.csv", index=False)
    
    # Save metadata
    metadata = {
        'pathology_classes': pathology_classes,
        'num_classes': len(pathology_classes),
        'min_occurrences': min_occurrences,
        'pathology_counts': {p: int(pathology_counts[p]) for p in pathology_classes},
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_patients': int(len(train_patients)),
        'val_patients': int(len(val_patients)),
        'test_patients': int(len(test_patients))
    }
    
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Files saved to: {output_dir}")
    print(f"   Total pathologies: {len(pathology_classes)}")
    print("=" * 80)
    
    return train_df, val_df, test_df, metadata, pathology_classes


if __name__ == "__main__":
    CSV_PATH = "data/raw/final_multimodal_dataset.csv"
    OUTPUT_DIR = "data/processed"
    
    train_df, val_df, test_df, metadata, pathologies = prepare_dataset(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        min_occurrences=20,  # Increased threshold
        test_size=0.15,
        val_size=0.15
    )
    
    print("\n📊 Dataset Summary:")
    print(f"   Training:   {len(train_df):5d} images")
    print(f"   Validation: {len(val_df):5d} images")
    print(f"   Testing:    {len(test_df):5d} images")
    print(f"   Total:      {len(train_df) + len(val_df) + len(test_df):5d} images")
    print(f"\n🏥 Discovered Pathologies: {len(pathologies)}")
    for p in pathologies:
        print(f"   - {p}")