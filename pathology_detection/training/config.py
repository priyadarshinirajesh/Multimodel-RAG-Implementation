# pathology_detection/training/config.py

from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    WEIGHTS_DIR = PROJECT_ROOT / "pathology_detection" / "weights"
    LOGS_DIR = PROJECT_ROOT / "pathology_detection" / "logs"
    
    # Create directories
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    TRAIN_CSV = DATA_DIR / "train_labels.csv"
    VAL_CSV = DATA_DIR / "val_labels.csv"
    TEST_CSV = DATA_DIR / "test_labels.csv"
    
    # Pathology classes
    PATHOLOGY_CLASSES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    NUM_CLASSES = len(PATHOLOGY_CLASSES)
    
    # Model architecture
    MODEL_NAME = "densenet121"  # Options: densenet121, resnet50, efficientnet_b0
    PRETRAINED = True
    FREEZE_BACKBONE = False  # Set to True for faster training with less data
    
    # Image preprocessing
    IMAGE_SIZE = 224  # Standard for ImageNet-pretrained models
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Training hyperparameters
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"  # Options: cosine, step, plateau
    LR_WARMUP_EPOCHS = 2
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.5  # 0.0 to 1.0
    
    # Loss function
    LOSS_FUNCTION = "bce"  # Binary Cross Entropy for multi-label
    FOCAL_LOSS_GAMMA = 2.0  # If using focal loss for imbalanced classes
    CLASS_WEIGHTS = None  # Will be computed from data if None
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Device
    DEVICE = "cuda"  # Will auto-detect in training script
    NUM_WORKERS = 4  # DataLoader workers
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    CHECKPOINT_METRIC = "val_auroc"  # Options: val_loss, val_auroc, val_f1
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    USE_TENSORBOARD = True
    
    # Evaluation
    EVAL_THRESHOLD = 0.5  # For binary classification
    EVAL_METRICS = ["auroc", "f1", "precision", "recall"]


# Create global config instance
cfg = Config()