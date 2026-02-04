# evaluation/generate_performance_report.py

"""
Generate comprehensive performance report for research paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

RESULTS_DIR = "pathology_detection/evaluation/results"

def generate_report():
    print("generating Final Performance Report...")
    
    # 1. Load Comparison Data
    try:
        df_base = pd.read_csv(f"{RESULTS_DIR}/results_baseline.csv")
        df_prop = pd.read_csv(f"{RESULTS_DIR}/results_proposed.csv")
        
        # Calculate improvements
        metrics = {
            'BLEU Score': 'bleu_score',
            'ROUGE-1': 'rouge1',
            'BERT Similarity': 'bert_similarity'
        }
        
        report_lines = []
        report_lines.append("# Research Paper Performance Summary")
        report_lines.append("\n## 1. Generative Performance Comparison")
        report_lines.append("| Metric | Baseline (Standard RAG) | Proposed (Pathology-Aware) | Improvement |")
        report_lines.append("|--------|-------------------------|----------------------------|-------------|")
        
        # Data for plotting
        plot_data = []
        
        for name, key in metrics.items():
            if key in df_prop.columns:
                base_val = df_base[key].mean()
                prop_val = df_prop[key].mean()
                imp = ((prop_val - base_val) / base_val) * 100
                
                report_lines.append(f"| {name} | {base_val:.4f} | {prop_val:.4f} | **{imp:+.2f}%** |")
                
                plot_data.append({'Metric': name, 'Score': base_val, 'System': 'Baseline'})
                plot_data.append({'Metric': name, 'Score': prop_val, 'System': 'Proposed'})

        # Generate Bar Chart
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_plot, x='Metric', y='Score', hue='System', palette=['#95a5a6', '#2ecc71'])
            plt.title("Performance Comparison: Standard vs. Pathology-Aware RAG")
            plt.ylim(0, 1.0)
            plt.savefig(f"{RESULTS_DIR}/comparison_chart.png")
            print(f"✅ Generated comparison chart: {RESULTS_DIR}/comparison_chart.png")

    except Exception as e:
        print(f"⚠️ Could not generate comparison report (Did you run the comparison script?): {e}")

    # 2. Load Classification Metrics (from training)
    try:
        # Assuming we saved a summary JSON during evaluation
        # If not, we can parse the CSV
        preds_path = f"{RESULTS_DIR}/test_predictions.csv"
        if os.path.exists(preds_path):
            report_lines.append("\n## 2. Pathology Detection Accuracy (DenseNet-121)")
            report_lines.append("Performance of the novel specialist agent on the test set:")
            # You can add logic here to summarize the test_predictions.csv if needed
            # For now, we assume the user has the 'roc_curves.png' generated previously
            report_lines.append("\n*See 'roc_curves.png' and 'confusion_matrices.png' for detailed visual analysis.*")
            
    except Exception as e:
        print(f"Error loading detection metrics: {e}")

    # 3. Save Markdown Report
    with open(f"{RESULTS_DIR}/FINAL_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"✅ Report saved to: {RESULTS_DIR}/FINAL_REPORT.md")
    print("\n" + "\n".join(report_lines))

if __name__ == "__main__":
    generate_report()