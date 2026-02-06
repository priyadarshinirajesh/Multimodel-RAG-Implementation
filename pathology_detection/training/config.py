# pathology_detection/training/config.py

from pathlib import Path
import json

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
    METADATA_FILE = DATA_DIR / "dataset_metadata.json"
    
    # ✅ NEW: Load pathology classes dynamically from metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
            PATHOLOGY_CLASSES = metadata['pathology_classes']
            NUM_CLASSES = len(PATHOLOGY_CLASSES)
            print(f"✅ Loaded {NUM_CLASSES} pathologies from metadata")
    else:
        # Fallback to NIH-14 if metadata doesn't exist yet
        print("⚠️  Metadata not found, using NIH-14 pathologies as fallback")
        PATHOLOGY_CLASSES = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        NUM_CLASSES = len(PATHOLOGY_CLASSES)
    
    # Model architecture
    MODEL_NAME = "densenet121"
    PRETRAINED = True
    FREEZE_BACKBONE = False
    
    # Image preprocessing
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"
    LR_WARMUP_EPOCHS = 2
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.5
    
    # Loss function
    LOSS_FUNCTION = "bce"
    FOCAL_LOSS_GAMMA = 2.0
    CLASS_WEIGHTS = None
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Device
    DEVICE = "cuda"
    NUM_WORKERS = 4
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    CHECKPOINT_METRIC = "val_auroc"
    
    # Logging
    LOG_INTERVAL = 10
    USE_TENSORBOARD = True
    
    # Evaluation
    EVAL_THRESHOLD = 0.5
    EVAL_METRICS = ["auroc", "f1", "precision", "recall"]


# Create global config instance
cfg = Config()