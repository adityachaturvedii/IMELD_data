"""
FLD Classification: Comprehensive Results Visualization
=======================================================
Author: Medical ML Research Assistant
Purpose: Generate publication-quality figures:
         - ROC and PR curves with confidence bands
         - Model performance comparison
         - Calibration plots
         - Feature importance aggregation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_results(filepath='results/nested_cv_results.pkl'):
    """Load saved modeling results."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_roc_curves(results):
    """Plot ROC curves for all models with confidence bands."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 'LightGBM': '#e74c3c'}

    for model_idx, model_results in enumerate(results['fold_results']):
        model_name = model_results['model_name']
        ax = axes[model_idx]

        # Collect ROC curves from all folds
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold_result in model_results['fold_results']:
            y_test = fold_result['y_test']
            y_proba = fold_result['y_proba']

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_curve(y_test, y_proba))

            # Plot individual fold (light)
            ax.plot(fpr, tpr, alpha=0.15, color=colors[model_name], linewidth=1)

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = model_results['aggregated_metrics']['auc_roc']['mean']
        std_auc = model_results['aggregated_metrics']['auc_roc']['std']

        ax.plot(mean_fpr, mean_tpr, color=colors[model_name], linewidth=3,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

        # Confidence band
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                        color=colors[model_name], alpha=0.2,
                        label='± 1 std. dev.')

        # Diagonal (chance)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Chance')

        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(f'{model_name} - ROC Curve\n5-Fold Nested CV', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig('results/02_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/02_roc_curves.png")
    plt.close()

def plot_pr_curves(results):
    """Plot Precision-Recall curves for all models."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 'LightGBM': '#e74c3c'}

    for model_idx, model_results in enumerate(results['fold_results']):
        model_name = model_results['model_name']
        ax = axes[model_idx]

        # Get baseline (prevalence)
        n_pos = results['data_info']['class_distribution']['positive']
        n_total = results['data_info']['n_samples']
        baseline = n_pos / n_total

        # Collect PR curves from all folds
        for fold_result in model_results['fold_results']:
            y_test = fold_result['y_test']
            y_proba = fold_result['y_proba']

            precision, recall, _ = precision_recall_curve(y_test, y_proba)

            # Plot individual fold (light)
            ax.plot(recall, precision, alpha=0.15, color=colors[model_name], linewidth=1)

        # Get mean PR-AUC
        mean_pr_auc = model_results['aggregated_metrics']['auc_pr']['mean']
        std_pr_auc = model_results['aggregated_metrics']['auc_pr']['std']

        # Baseline (no-skill)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'Baseline (prevalence = {baseline:.3f})')

        # Add text annotation for mean PR-AUC
        ax.text(0.5, 0.05, f'Mean PR-AUC = {mean_pr_auc:.3f} ± {std_pr_auc:.3f}',
               fontsize=11, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=colors[model_name], alpha=0.3))

        ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontweight='bold')
        ax.set_title(f'{model_name} - Precision-Recall Curve\n5-Fold Nested CV',
                    fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig('results/03_pr_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/03_pr_curves.png")
    plt.close()

def plot_model_comparison(results):
    """Compare models across all metrics."""

    # Extract metrics
    models = []
    metrics_data = {
        'AUC-ROC': [],
        'PR-AUC': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'Specificity': []
    }
    errors = {
        'AUC-ROC': [],
        'PR-AUC': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'Specificity': []
    }

    for model_results in results['fold_results']:
        model_name = model_results['model_name']
        models.append(model_name)

        agg = model_results['aggregated_metrics']
        metrics_data['AUC-ROC'].append(agg['auc_roc']['mean'])
        metrics_data['PR-AUC'].append(agg['auc_pr']['mean'])
        metrics_data['F1'].append(agg['f1']['mean'])
        metrics_data['Precision'].append(agg['precision']['mean'])
        metrics_data['Recall'].append(agg['recall']['mean'])
        metrics_data['Specificity'].append(agg['specificity']['mean'])

        errors['AUC-ROC'].append(agg['auc_roc']['ci_95'])
        errors['PR-AUC'].append(agg['auc_pr']['ci_95'])
        errors['F1'].append(agg['f1']['ci_95'])
        errors['Precision'].append(agg['precision']['ci_95'])
        errors['Recall'].append(agg['recall']['ci_95'])
        errors['Specificity'].append(agg['specificity']['ci_95'])

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]

        x = np.arange(len(models))
        bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(x, values, yerr=list(errors[metric_name]), fmt='none',
                   ecolor='black', capsize=5, capthick=2, alpha=0.8)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + errors[metric_name][i] + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_ylabel(metric_name, fontweight='bold', fontsize=11)
        ax.set_title(f'{metric_name} Comparison\n(5-Fold Nested CV)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best model
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('results/04_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/04_model_comparison.png")
    plt.close()

def plot_calibration(results):
    """Plot calibration curves to assess probability reliability."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 'LightGBM': '#e74c3c'}

    for model_idx, model_results in enumerate(results['fold_results']):
        model_name = model_results['model_name']
        ax = axes[model_idx]

        # Collect all predictions across folds
        y_true_all = []
        y_proba_all = []

        for fold_result in model_results['fold_results']:
            y_true_all.extend(fold_result['y_test'])
            y_proba_all.extend(fold_result['y_proba'])

        y_true_all = np.array(y_true_all)
        y_proba_all = np.array(y_proba_all)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true_all, y_proba_all, n_bins=10, strategy='uniform')

        # Plot
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
               color=colors[model_name], label=f'{model_name}')

        # Perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect calibration')

        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontweight='bold')
        ax.set_title(f'{model_name} - Calibration Curve', fontweight='bold', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig('results/05_calibration_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/05_calibration_curves.png")
    plt.close()

def create_metrics_table(results):
    """Create comprehensive metrics table."""

    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS TABLE")
    print("="*80)

    rows = []
    for model_results in results['fold_results']:
        model_name = model_results['model_name']
        agg = model_results['aggregated_metrics']

        row = {
            'Model': model_name,
            'AUC-ROC': f"{agg['auc_roc']['mean']:.4f} ± {agg['auc_roc']['std']:.4f}",
            'PR-AUC': f"{agg['auc_pr']['mean']:.4f} ± {agg['auc_pr']['std']:.4f}",
            'F1': f"{agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}",
            'Precision': f"{agg['precision']['mean']:.4f} ± {agg['precision']['std']:.4f}",
            'Recall': f"{agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}",
            'Specificity': f"{agg['specificity']['mean']:.4f} ± {agg['specificity']['std']:.4f}",
            'Accuracy': f"{agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('results/metrics_summary.csv', index=False)
    print("\n✓ Saved: results/metrics_summary.csv")

    # Identify best model
    best_auc = max([model_results['aggregated_metrics']['auc_roc']['mean']
                   for model_results in results['fold_results']])
    best_model = [model_results['model_name']
                 for model_results in results['fold_results']
                 if model_results['aggregated_metrics']['auc_roc']['mean'] == best_auc][0]

    print(f"\n🏆 BEST MODEL (by AUC-ROC): {best_model} (AUC = {best_auc:.4f})")

    return df

def plot_fold_variability(results):
    """Visualize performance variability across folds."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 'LightGBM': '#e74c3c'}
    metrics = ['auc_roc', 'auc_pr', 'f1', 'recall']
    titles = ['AUC-ROC', 'PR-AUC', 'F1 Score', 'Recall (Sensitivity)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        for model_results in results['fold_results']:
            model_name = model_results['model_name']
            values = model_results['aggregated_metrics'][metric]['values']

            # Box plot
            positions = [idx for idx, m in enumerate(results['fold_results']) if m['model_name'] == model_name]
            bp = ax.boxplot([values], positions=positions, widths=0.6,
                           patch_artist=True, notch=True,
                           boxprops=dict(facecolor=colors[model_name], alpha=0.7),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5))

            # Individual points
            x = np.random.normal(positions[0], 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.6, color=colors[model_name], s=50, edgecolors='black', linewidth=1)

        ax.set_ylabel(title, fontweight='bold', fontsize=11)
        ax.set_title(f'{title} - Fold Variability', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(results['fold_results'])))
        ax.set_xticklabels([m['model_name'] for m in results['fold_results']], fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/06_fold_variability.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/06_fold_variability.png")
    plt.close()

if __name__ == "__main__":

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Load results
    print("\nLoading results...")
    results = load_results()
    print("✓ Results loaded")

    # Generate all plots
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_model_comparison(results)
    plot_calibration(results)
    plot_fold_variability(results)

    # Create metrics table
    create_metrics_table(results)

    print("\n" + "="*80)
    print("✓ All visualizations generated successfully!")
    print("="*80)
