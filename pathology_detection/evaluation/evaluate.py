# pathology_detection/evaluation/evaluate.py

"""
Evaluation script for trained pathology detection model
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
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
                
                print(f"{pathology:35s} | AUROC: {auroc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Support: {support}")
            else:
                print(f"{pathology:35s} | Insufficient samples for evaluation")
        
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
        print(f"{'MACRO AVERAGE':35s} | AUROC: {macro_auroc:.4f} | F1: {macro_f1:.4f}")
        print(f"{'WEIGHTED AVERAGE':35s} | AUROC: {weighted_auroc:.4f} | F1: {weighted_f1:.4f}")
        print("="*80)
        
        summary = {
            'per_class': results,
            'macro_auroc': float(macro_auroc),
            'macro_f1': float(macro_f1),
            'weighted_auroc': float(weighted_auroc),
            'weighted_f1': float(weighted_f1)
        }
        
        return summary, results_df
    
    def plot_single_aggregated_roc_curve(self, predictions, labels, save_dir):
        """
        Plot SINGLE aggregated ROC curve for overall model performance
        Uses micro-averaging (treats all predictions as one large binary classification)
        """
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("   Creating single aggregated ROC curve...")
        
        # METHOD 1: Micro-averaging (flatten all predictions and labels)
        # This treats the entire multi-label problem as one big binary classification
        y_true_flat = labels.ravel()  # Flatten all labels
        y_pred_flat = predictions.ravel()  # Flatten all predictions
        
        # Calculate micro-averaged ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, y_pred_flat)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # METHOD 2: Macro-averaging (average of all individual ROC curves)
        # First, interpolate all ROC curves to the same points
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for i in range(len(self.cfg.PATHOLOGY_CLASSES)):
            if len(np.unique(labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc_score(labels[:, i], predictions[:, i]))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        roc_auc_macro = auc(mean_fpr, mean_tpr)
        
        # Calculate standard deviation for confidence interval
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot micro-averaged ROC
        ax.plot(fpr_micro, tpr_micro, 
                color='deepskyblue', lw=3, 
                label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')
        
        # Plot macro-averaged ROC with confidence interval
        ax.plot(mean_fpr, mean_tpr, 
                color='navy', lw=3, 
                label=f'Macro-average ROC (AUC = {roc_auc_macro:.3f})')
        
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                        color='navy', alpha=0.2, 
                        label='± 1 std. dev.')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('Overall ROC Curve - Pathology Detection Model', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text box with additional info
        info_text = f'Evaluated on {len(self.cfg.PATHOLOGY_CLASSES)} pathologies\n'
        info_text += f'Total test samples: {len(labels)}'
        
        ax.text(0.98, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'single_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved single ROC curve to: {save_dir / 'single_roc_curve.png'}")
        print(f"   Micro-average AUC: {roc_auc_micro:.4f}")
        print(f"   Macro-average AUC: {roc_auc_macro:.4f}")
        plt.close()
        
        return roc_auc_micro, roc_auc_macro
    
    def plot_combined_roc_curve(self, predictions, labels, save_dir):
        """
        Plot SINGLE combined ROC curve showing all pathologies with different colors
        """
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("   Creating combined ROC curve...")
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get colormap for different pathologies
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(self.cfg.PATHOLOGY_CLASSES)))
        
        auroc_scores = []
        
        # Plot ROC curve for each pathology
        for i, (pathology, color) in enumerate(zip(self.cfg.PATHOLOGY_CLASSES, colors)):
            if len(np.unique(labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
                auroc_scores.append(roc_auc)
                
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{pathology} (AUC = {roc_auc:.3f})',
                        alpha=0.8)
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        # Calculate and plot mean ROC
        mean_auc = np.mean(auroc_scores)
        plt.plot([], [], ' ', label=f'\nMean AUC = {mean_auc:.3f}')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curves - All Pathologies', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'combined_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved combined ROC curve to: {save_dir / 'combined_roc_curve.png'}")
        plt.close()
    
    def plot_overall_confusion_matrix(self, predictions, labels, save_dir, threshold=0.5):
        """
        Plot SINGLE combined confusion matrix aggregating all pathologies
        """
        
        save_dir = Path(save_dir)
        binary_preds = (predictions >= threshold).astype(int)
        
        print("   Creating overall confusion matrix...")
        
        # Aggregate confusion matrix across all pathologies
        total_tn = 0
        total_fp = 0
        total_fn = 0
        total_tp = 0
        
        for i in range(len(self.cfg.PATHOLOGY_CLASSES)):
            if len(np.unique(labels[:, i])) > 1:
                tn, fp, fn, tp = confusion_matrix(labels[:, i], binary_preds[:, i]).ravel()
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp
        
        # Create aggregated confusion matrix
        overall_cm = np.array([[total_tn, total_fp],
                              [total_fn, total_tp]])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            overall_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['True Negative', 'True Positive'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'}
        )
        
        plt.title('Overall Confusion Matrix (Aggregated Across All Pathologies)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Actual', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=14, fontweight='bold')
        
        # Add accuracy and other metrics as text
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        plt.text(2.3, 0.5, metrics_text, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'overall_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved overall confusion matrix to: {save_dir / 'overall_confusion_matrix.png'}")
        plt.close()
    
    def plot_top_pathologies_comparison(self, predictions, labels, save_dir, top_n=10):
        """
        Plot bar chart comparing AUROC scores for top N pathologies
        """
        
        save_dir = Path(save_dir)
        
        print(f"   Creating top {top_n} pathologies comparison...")
        
        # Calculate AUROC for each pathology
        auroc_data = []
        
        for i, pathology in enumerate(self.cfg.PATHOLOGY_CLASSES):
            if len(np.unique(labels[:, i])) > 1:
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
                support = int(labels[:, i].sum())
                auroc_data.append({
                    'Pathology': pathology,
                    'AUROC': auroc,
                    'Support': support
                })
        
        # Sort by AUROC and get top N
        auroc_df = pd.DataFrame(auroc_data)
        top_pathologies = auroc_df.nlargest(top_n, 'AUROC')
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#2ecc71' if x >= 0.8 else '#f39c12' if x >= 0.7 else '#e74c3c' 
                 for x in top_pathologies['AUROC']]
        
        bars = ax.barh(top_pathologies['Pathology'], top_pathologies['AUROC'], color=colors)
        
        # Add value labels on bars
        for i, (bar, auroc, support) in enumerate(zip(bars, top_pathologies['AUROC'], top_pathologies['Support'])):
            ax.text(auroc + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{auroc:.3f} (n={support})', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('AUROC Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Top {top_n} Pathologies by AUROC', fontsize=16, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.8)')
        ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.7)')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'top_pathologies_auroc.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved top pathologies comparison to: {save_dir / 'top_pathologies_auroc.png'}")
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
        
        # ✅ Single aggregated ROC curve (what you want!)
        micro_auc, macro_auc = self.plot_single_aggregated_roc_curve(predictions, labels, output_dir)
        
        # Overall confusion matrix
        self.plot_overall_confusion_matrix(predictions, labels, output_dir)
        
        # Top pathologies bar chart
        self.plot_top_pathologies_comparison(predictions, labels, output_dir, top_n=15)
        
        # 5. Save predictions
        print("\n[5/5] Saving detailed predictions...")
        self.save_predictions(predictions, labels, metadata, output_dir)
        
        # Add AUC values to summary
        summary['micro_average_auc'] = float(micro_auc)
        summary['macro_average_auc'] = float(macro_auc)
        
        # Re-save summary with AUC values
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
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