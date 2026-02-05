# evaluation/generate_performance_report.py

"""
Generate comprehensive performance report for research paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def generate_report(
    pathology_metrics_file="pathology_detection/evaluation/results/per_class_metrics.csv",
    system_results_file="pathology_comparison_results.csv"
):
    """
    Generate a comprehensive performance report
    """
    
    print("="*80)
    print("GENERATING PERFORMANCE REPORT")
    print("="*80)
    
    # Load pathology detection metrics
    pathology_df = pd.read_csv(pathology_metrics_file)
    
    # Load system-level results
    system_df = pd.read_csv(system_results_file)
    
    # Create report
    report = {
        "pathology_detection_performance": {
            "mean_auroc": float(pathology_df["AUROC"].mean()),
            "mean_f1": float(pathology_df["F1"].mean()),
            "mean_precision": float(pathology_df["Precision"].mean()),
            "mean_recall": float(pathology_df["Recall"].mean()),
            "best_performing_pathology": pathology_df.loc[pathology_df["AUROC"].idxmax(), "Pathology"],
            "best_auroc": float(pathology_df["AUROC"].max())
        },
        "system_performance": {
            "mean_clinical_correctness": float(system_df["clinical_correctness"].mean()),
            "mean_groundedness": float(system_df["groundedness"].mean()),
            "mean_completeness": float(system_df["completeness"].mean()),
            "pathology_detection_usage_rate": float(system_df["used_pathology_detection"].mean())
        },
        "top_5_pathologies": pathology_df.nlargest(5, "AUROC")[
            ["Pathology", "AUROC", "F1", "Support"]
        ].to_dict("records")
    }
    
    # Save report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n📊 PATHOLOGY DETECTION PERFORMANCE:")
    print(f"  Mean AUROC:     {report['pathology_detection_performance']['mean_auroc']:.4f}")
    print(f"  Mean F1:        {report['pathology_detection_performance']['mean_f1']:.4f}")
    print(f"  Best Pathology: {report['pathology_detection_performance']['best_performing_pathology']} ({report['pathology_detection_performance']['best_auroc']:.4f})")
    
    print("\n🎯 SYSTEM PERFORMANCE:")
    print(f"  Clinical Correctness: {report['system_performance']['mean_clinical_correctness']:.4f}")
    print(f"  Groundedness:         {report['system_performance']['mean_groundedness']:.4f}")
    print(f"  Completeness:         {report['system_performance']['mean_completeness']:.4f}")
    
    print("\n✅ Report saved to: performance_report.json")
    
    # Create visualizations
    create_visualizations(pathology_df, system_df)
    
    return report


def create_visualizations(pathology_df, system_df):
    """Create publication-ready visualizations"""
    
    # Figure 1: Pathology Detection Performance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top pathologies by AUROC
    top_pathologies = pathology_df.nlargest(10, "AUROC")
    axes[0].barh(top_pathologies["Pathology"], top_pathologies["AUROC"], color='steelblue')
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Top 10 Pathologies by AUROC")
    axes[0].grid(axis='x', alpha=0.3)
    
    # Metrics comparison
    metrics = ["AUROC", "F1", "Precision", "Recall"]
    mean_values = [pathology_df[m].mean() for m in metrics]
    axes[1].bar(metrics, mean_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_ylabel("Score")
    axes[1].set_title("Average Pathology Detection Metrics")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pathology_performance.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: pathology_performance.png")
    plt.close()
    
    # Figure 2: System Metrics Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    system_metrics = system_df[["clinical_correctness", "groundedness", "completeness"]]
    system_metrics.boxplot(ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("System Performance Metrics Distribution")
    ax.set_xticklabels(["Clinical Correctness", "Groundedness", "Completeness"])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("system_performance.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: system_performance.png")
    plt.close()


if __name__ == "__main__":
    generate_report()