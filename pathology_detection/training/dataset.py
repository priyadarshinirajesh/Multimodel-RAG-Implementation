# pathology_detection/training/dataset.py

"""
PyTorch Dataset for chest X-ray pathology detection
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChestXrayDataset(Dataset):
    """
    Dataset class for multi-label chest X-ray classification
    """
    
    def __init__(
        self,
        csv_file: str,
        pathology_classes: list,
        transform=None,
        is_training: bool = True
    ):
        """
        Args:
            csv_file: Path to CSV with labels
            pathology_classes: List of pathology names
            transform: Albumentations transform
            is_training: Whether this is training set
        """
        
        self.df = pd.read_csv(csv_file)
        self.pathology_classes = pathology_classes
        self.transform = transform
        self.is_training = is_training
        
        # Label columns
        self.label_columns = [f'label_{p}' for p in pathology_classes]
        
        print(f"Loaded {len(self.df)} images from {csv_file}")
        print(f"Mode: {'Training' if is_training else 'Validation/Test'}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, 224, 224]
            labels: Tensor [num_classes]
            metadata: dict with additional info
        """
        
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['filename']
        
        # Handle both absolute and relative paths
        if not Path(image_path).exists():
            # Try relative path from project root
            project_root = Path(__file__).parent.parent.parent
            image_path = project_root / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Extract labels
        labels = row[self.label_columns].values.astype(np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Metadata
        metadata = {
            'patient_id': row['patient_id'],
            'uid': row['uid'],
            'projection': row.get('projection', 'Unknown'),
            'image_path': str(image_path)
        }
        
        return image, torch.from_numpy(labels), metadata


def get_train_transforms(image_size=224):
    """
    Augmentation transforms for training
    """
    
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Add slight noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Normalization (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    """
    Transforms for validation/testing (no augmentation)
    """
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_dataloaders(cfg):
    """
    Create train, val, test dataloaders
    
    Args:
        cfg: Config object
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Transforms
    train_transform = get_train_transforms(cfg.IMAGE_SIZE)
    val_transform = get_val_transforms(cfg.IMAGE_SIZE)
    
    # Datasets
    train_dataset = ChestXrayDataset(
        csv_file=cfg.TRAIN_CSV,
        pathology_classes=cfg.PATHOLOGY_CLASSES,
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = ChestXrayDataset(
        csv_file=cfg.VAL_CSV,
        pathology_classes=cfg.PATHOLOGY_CLASSES,
        transform=val_transform,
        is_training=False
    )
    
    test_dataset = ChestXrayDataset(
        csv_file=cfg.TEST_CSV,
        pathology_classes=cfg.PATHOLOGY_CLASSES,
        transform=val_transform,
        is_training=False
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("DataLoaders Created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    from config import cfg
    
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # Get a batch
    images, labels, metadata = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"\nSample labels:\n{labels[0]}")
    print(f"\nSample metadata:\n{metadata}")