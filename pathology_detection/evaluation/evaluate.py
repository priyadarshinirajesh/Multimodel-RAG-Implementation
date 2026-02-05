# pathology_detection/evaluation/evaluate.py

"""
Evaluation script for trained pathology detection model
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pathology_detection.models.densenet_classifier import build_model
from pathology_detection.training.dataset import create_dataloaders
from pathology_detection.training.config import cfg


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, checkpoint_path, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = build_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Loaded model from: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val AUROC: {checkpoint['metrics']['auroc']:.4f}")
        
        # Get dataloaders
        _, _, self.test_loader = create_dataloaders(config)
    
    def predict(self):
        """
        Run prediction on test set
        
        Returns:
            predictions: numpy array [N, num_classes]
            labels: numpy array [N, num_classes]
            metadata: list of dicts
        """
        
        all_preds = []
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.test_loader, desc="Predicting"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_preds.append(probs)
                all_labels.append(labels.numpy())
                all_metadata.extend([
                    {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] 
                     for k, v in metadata.items()}
                    for i in range(len(labels))
                ])
        
        predictions = np.vstack(all_preds)
        labels = np.vstack(all_labels)
        
        return predictions, labels, all_metadata
    
    def evaluate_metrics(self, predictions, labels, threshold=0.5):
        """
        Calculate comprehensive metrics
        """
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # Binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        
        # Per-class metrics
        results = []
        
        for i, pathology in enumerate(self.cfg.PATHOLOGY_CLASSES):
            if len(np.unique(labels[:, i])) > 1:
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
                f1 = f1_score(labels[:, i], binary_preds[:, i], zero_division=0)
                precision = precision_score(labels[:, i], binary_preds[:, i], zero_division=0)
                recall = recall_score(labels[:, i], binary_preds[:, i], zero_division=0)
                
                # Support (number of positive samples)
                support = int(labels[:, i].sum())
                
                results.append({
                    'Pathology': pathology,
                    'AUROC': auroc,
                    'F1': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'Support': support
                })
                
                print(f"{pathology:20s} | AUROC: {auroc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Support: {support}")
            else:
                print(f"{pathology:20s} | Insufficient samples for evaluation")
        
        # Overall metrics
        results_df = pd.DataFrame(results)
        
        # Weighted average (by support)
        total_support = results_df['Support'].sum()
        weighted_auroc = (results_df['AUROC'] * results_df['Support']).sum() / total_support
        weighted_f1 = (results_df['F1'] * results_df['Support']).sum() / total_support
        
        # Macro average
        macro_auroc = results_df['AUROC'].mean()
        macro_f1 = results_df['F1'].mean()
        
        print("\n" + "-"*80)
        print(f"{'MACRO AVERAGE':20s} | AUROC: {macro_auroc:.4f} | F1: {macro_f1:.4f}")
        print(f"{'WEIGHTED AVERAGE':20s} | AUROC: {weighted_auroc:.4f} | F1: {weighted_f1:.4f}")
        print("="*80)
        
        summary = {
            'per_class': results,
            'macro_auroc': float(macro_auroc),
            'macro_f1': float(macro_f1),
            'weighted_auroc': float(weighted_auroc),
            'weighted_f1': float(weighted_f1)
        }
        
        return summary, results_df
    
    def plot_roc_curves(self, predictions, labels, save_dir):
        """
        Plot ROC curves for each pathology
        """
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        for i, pathology in enumerate(self.cfg.PATHOLOGY_CLASSES):
            if len(np.unique(labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
                
                axes[i].plot(fpr, tpr, label=f'AUROC = {auroc:.3f}', linewidth=2)
                axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(pathology)
                axes[i].legend(loc='lower right')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.cfg.PATHOLOGY_CLASSES), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved ROC curves to: {save_dir / 'roc_curves.png'}")
        plt.close()
    
    def plot_confusion_matrices(self, predictions, labels, save_dir, threshold=0.5):
        """
        Plot confusion matrices for top pathologies
        """
        
        save_dir = Path(save_dir)
        binary_preds = (predictions >= threshold).astype(int)
        
        # Select top 6 pathologies by support
        support_counts = labels.sum(axis=0)
        top_indices = np.argsort(support_counts)[::-1][:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, i in enumerate(top_indices):
            pathology = self.cfg.PATHOLOGY_CLASSES[i]
            cm = confusion_matrix(labels[:, i], binary_preds[:, i])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[idx]
            )
            axes[idx].set_title(pathology)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved confusion matrices to: {save_dir / 'confusion_matrices.png'}")
        plt.close()
    
    def save_predictions(self, predictions, labels, metadata, save_dir):
        """
        Save predictions to CSV for analysis
        """
        
        save_dir = Path(save_dir)
        
        # Create DataFrame
        results = []
        
        for i, meta in enumerate(metadata):
            row = {
                'patient_id': meta['patient_id'],
                'uid': meta['uid'],
                'projection': meta['projection'],
                'image_path': meta['image_path']
            }
            
            # Add predictions and ground truth for each pathology
            for j, pathology in enumerate(self.cfg.PATHOLOGY_CLASSES):
                row[f'pred_{pathology}'] = predictions[i, j]
                row[f'true_{pathology}'] = labels[i, j]
            
            results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_dir / 'test_predictions.csv', index=False)
        print(f"✅ Saved predictions to: {save_dir / 'test_predictions.csv'}")
        
        return results_df
    
    def run_full_evaluation(self, output_dir="pathology_detection/evaluation/results"):
        """
        Run complete evaluation pipeline
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("RUNNING FULL EVALUATION")
        print("="*80)
        
        # 1. Get predictions
        print("\n[1/5] Generating predictions...")
        predictions, labels, metadata = self.predict()
        
        # 2. Calculate metrics
        print("\n[2/5] Calculating metrics...")
        summary, results_df = self.evaluate_metrics(predictions, labels)
        
        # 3. Save metrics
        print("\n[3/5] Saving metrics...")
        results_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
        
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 4. Plot visualizations
        print("\n[4/5] Creating visualizations...")
        self.plot_roc_curves(predictions, labels, output_dir)
        self.plot_confusion_matrices(predictions, labels, output_dir)
        
        # 5. Save predictions
        print("\n[5/5] Saving detailed predictions...")
        self.save_predictions(predictions, labels, metadata, output_dir)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
        return summary


def main():
    """Main entry point"""
    
    # Path to best model checkpoint
    checkpoint_path = cfg.WEIGHTS_DIR / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first!")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(checkpoint_path, cfg)
    
    # Run evaluation
    summary = evaluator.run_full_evaluation()
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()