# pathology_detection/training/train.py
"""
Training script for pathology detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

# Import from our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pathology_detection.models.densenet_classifier import build_model
from pathology_detection.training.dataset import create_dataloaders
from pathology_detection.training.config import cfg
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


class Trainer:
    """
    Training manager for pathology detection
    """
    
    def __init__(self, config):
        self.cfg = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model
        self.model = build_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED,
            freeze_backbone=config.FREEZE_BACKBONE
        ).to(self.device)
        
        # Data
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        elif config.LR_SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.best_val_auroc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aurocs = []
        
        # Tensorboard
        if config.USE_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(
                log_dir=config.LOGS_DIR / f"run_{timestamp}"
            )
        else:
            self.writer = None
        
        # Early stopping
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [Train]"
        )
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.cfg.LOG_INTERVAL == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate AUROC (per class)
        aurocs = []
        for i in range(self.cfg.NUM_CLASSES):
            if len(np.unique(all_labels[:, i])) > 1:  # Check if both classes present
                auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aurocs.append(auroc)
        
        mean_auroc = np.mean(aurocs) if aurocs else 0.0
        
        # Log epoch metrics
        if self.writer:
            self.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            self.writer.add_scalar('train/epoch_auroc', mean_auroc, epoch)
        
        self.train_losses.append(epoch_loss)
        
        return epoch_loss, mean_auroc
    
    def validate(self, epoch):
        """Validate model"""
        
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, 
                desc=f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [Val]  "
            )
            
            for images, labels, metadata in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Validation metrics
        val_loss = running_loss / len(self.val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics per class
        aurocs = []
        for i in range(self.cfg.NUM_CLASSES):
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aurocs.append(auroc)
            else:
                aurocs.append(0.5)  # Default for single-class
        
        mean_auroc = np.mean(aurocs)
        
        # Binary predictions for F1, precision, recall
        binary_preds = (all_preds >= 0.5).astype(int)
        
        # Calculate per-class metrics
        f1_scores = []
        precisions = []
        recalls = []
        
        for i in range(self.cfg.NUM_CLASSES):
            if np.sum(all_labels[:, i]) > 0:  # If class has positive samples
                f1 = f1_score(all_labels[:, i], binary_preds[:, i], zero_division=0)
                prec = precision_score(all_labels[:, i], binary_preds[:, i], zero_division=0)
                rec = recall_score(all_labels[:, i], binary_preds[:, i], zero_division=0)
                
                f1_scores.append(f1)
                precisions.append(prec)
                recalls.append(rec)
        
        mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
        mean_precision = np.mean(precisions) if precisions else 0.0
        mean_recall = np.mean(recalls) if recalls else 0.0
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/auroc', mean_auroc, epoch)
            self.writer.add_scalar('val/f1', mean_f1, epoch)
            self.writer.add_scalar('val/precision', mean_precision, epoch)
            self.writer.add_scalar('val/recall', mean_recall, epoch)
            
            # Log per-class AUROCs
            for i, (pathology, auroc) in enumerate(zip(self.cfg.PATHOLOGY_CLASSES, aurocs)):
                self.writer.add_scalar(f'val/auroc_{pathology}', auroc, epoch)
        
        self.val_losses.append(val_loss)
        self.val_aurocs.append(mean_auroc)
        
        metrics = {
            'loss': val_loss,
            'auroc': mean_auroc,
            'f1': mean_f1,
            'precision': mean_precision,
            'recall': mean_recall,
            'per_class_auroc': dict(zip(self.cfg.PATHOLOGY_CLASSES, aurocs))
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.cfg)
        }
        
        # Save latest
        checkpoint_path = self.cfg.WEIGHTS_DIR / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.cfg.WEIGHTS_DIR / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model with AUROC: {metrics['auroc']:.4f}")
    
    def train(self):
        """Main training loop"""
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_auroc = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
                if self.writer:
                    self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Check if best model
            is_best = val_metrics['auroc'] > self.best_val_auroc
            if is_best:
                self.best_val_auroc = val_metrics['auroc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print(f"Best Validation AUROC: {self.best_val_auroc:.4f}")
        print("="*80)
        
        if self.writer:
            self.writer.close()


def main():
    """Main entry point"""
    
    # Print configuration
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Epochs: {cfg.NUM_EPOCHS}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Number of Classes: {cfg.NUM_CLASSES}")
    print("="*80)
    
    # Create trainer
    trainer = Trainer(cfg)
    
    # Start training
    trainer.train()
    
    print("\n✅ Training script completed successfully!")


if __name__ == "__main__":
    main()