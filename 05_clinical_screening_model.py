"""
FLD Classification: Clinically-Realistic Screening Model
========================================================
Author: Medical ML Research Assistant
Purpose: Build practical screening model EXCLUDING CAPDbm
         - CAPDbm is a diagnostic tool (FibroScan), not available for population screening
         - Focus on readily-available clinical/anthropometric predictors
         - Compare diagnostic (with CAPDbm) vs. screening (without CAPDbm) performance
         - Assess clinical utility for primary care
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

CONFIG = {
    'data_path': 'non_invasive_clean.csv',
    'target': 'fld',
    'exclude_features_diagnostic': ['id', 'fld'],  # Include CAPDbm
    'exclude_features_screening': ['id', 'fld', 'CAPDbm'],  # Exclude CAPDbm for screening

    'outer_folds': 5,
    'inner_folds': 3,
    'random_state': 42,
    'n_jobs': -1,
}

def train_screening_models():
    """
    Train models with and without CAPDbm to demonstrate clinical vs. screening performance.
    """

    print("="*80)
    print("CLINICAL SCREENING MODEL (EXCLUDING CAPDbm)")
    print("="*80)
    print("\nRATIONALE:")
    print("  - CAPDbm is a FibroScan diagnostic measure, not available in primary care")
    print("  - Including it gives 'perfect' AUC but no clinical utility for screening")
    print("  - Screening model uses only readily-available predictors:")
    print("    → BMI, waist circumference, blood pressure, age, comorbidities")
    print("="*80)

    # Load data
    df = pd.read_csv(CONFIG['data_path'])
    print(f"\n✓ Loaded {df.shape[0]} samples")

    # Calculate class imbalance
    y = df[CONFIG['target']].values
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"✓ Class distribution: {n_neg} negative, {n_pos} positive")
    print(f"✓ Scale pos weight: {scale_pos_weight:.3f}")

    # Train both versions
    results = {}

    for model_type in ['diagnostic', 'screening']:

        print(f"\n{'='*80}")
        print(f"TRAINING: {model_type.upper()} MODEL")
        print(f"{'='*80}")

        # Select features
        if model_type == 'diagnostic':
            feature_cols = [col for col in df.columns
                          if col not in CONFIG['exclude_features_diagnostic']]
        else:
            feature_cols = [col for col in df.columns
                          if col not in CONFIG['exclude_features_screening']]

        X = df[feature_cols].values
        print(f"Features ({len(feature_cols)}): {feature_cols}")

        # Nested CV
        outer_cv = StratifiedKFold(n_splits=CONFIG['outer_folds'], shuffle=True,
                                  random_state=CONFIG['random_state'])

        fold_results = []
        all_shap_values = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            print(f"\n--- Fold {fold_idx}/{CONFIG['outer_folds']} ---")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Preprocessing (independent per fold)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=CONFIG['inner_folds'], shuffle=True,
                                      random_state=CONFIG['random_state'])

            # Use XGBoost as the model (best for tabular data)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 5],
                'scale_pos_weight': [scale_pos_weight],
                'random_state': [42]
            }

            model = xgb.XGBClassifier(eval_metric='logloss')
            grid_search = GridSearchCV(model, param_grid, cv=inner_cv,
                                      scoring='roc_auc', n_jobs=CONFIG['n_jobs'],
                                      verbose=0, refit=True)

            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_

            print(f"✓ Best inner CV AUC: {grid_search.best_score_:.4f}")

            # Predict
            y_pred = best_model.predict(X_test_scaled)
            y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            metrics = {
                'auc_roc': roc_auc_score(y_test, y_proba),
                'auc_pr': average_precision_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, y_pred)
            }

            print(f"✓ Test AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"✓ Test PR-AUC:  {metrics['auc_pr']:.4f}")
            print(f"✓ Test F1:      {metrics['f1']:.4f}")

            fold_results.append({
                'fold': fold_idx,
                'metrics': metrics,
                'y_test': y_test,
                'y_proba': y_proba
            })

            # SHAP values
            try:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_test_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                all_shap_values.append({
                    'shap_values': shap_values,
                    'X_test': X_test_scaled
                })
            except:
                pass

        # Aggregate metrics
        print(f"\n{'-'*80}")
        print(f"AGGREGATED RESULTS ({model_type.upper()})")
        print(f"{'-'*80}")

        aggregated = {}
        for metric_name in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']:
            values = [fold['metrics'][metric_name] for fold in fold_results]
            mean = np.mean(values)
            std = np.std(values)
            ci_95 = 1.96 * std / np.sqrt(len(values))

            aggregated[metric_name] = {'mean': mean, 'std': std, 'ci_95': ci_95, 'values': values}
            print(f"{metric_name.upper():12s}: {mean:.4f} ± {std:.4f} "
                  f"(95% CI: [{mean-ci_95:.4f}, {mean+ci_95:.4f}])")

        # Compute SHAP importance
        if all_shap_values:
            all_shap_concat = np.concatenate([s['shap_values'] for s in all_shap_values], axis=0)
            mean_abs_shap = np.abs(all_shap_concat).mean(axis=0)
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Mean_|SHAP|': mean_abs_shap
            }).sort_values('Mean_|SHAP|', ascending=False)
        else:
            feature_importance = None

        results[model_type] = {
            'fold_results': fold_results,
            'aggregated': aggregated,
            'feature_names': feature_cols,
            'feature_importance': feature_importance,
            'shap_values': all_shap_values
        }

    return results

def compare_diagnostic_vs_screening(results):
    """Compare performance with and without CAPDbm."""

    print("\n" + "="*80)
    print("DIAGNOSTIC vs. SCREENING MODEL COMPARISON")
    print("="*80)

    # Extract performance
    diag_auc = results['diagnostic']['aggregated']['auc_roc']['mean']
    screen_auc = results['screening']['aggregated']['auc_roc']['mean']

    diag_pr = results['diagnostic']['aggregated']['auc_pr']['mean']
    screen_pr = results['screening']['aggregated']['auc_pr']['mean']

    print(f"\nDIAGNOSTIC MODEL (with CAPDbm):")
    print(f"  AUC-ROC: {diag_auc:.4f}")
    print(f"  PR-AUC:  {diag_pr:.4f}")

    print(f"\nSCREENING MODEL (without CAPDbm - clinically realistic):")
    print(f"  AUC-ROC: {screen_auc:.4f}")
    print(f"  PR-AUC:  {screen_pr:.4f}")

    print(f"\nPERFORMANCE DROP:")
    print(f"  ΔAUC-ROC: {diag_auc - screen_auc:.4f} ({100*(diag_auc - screen_auc)/diag_auc:.1f}% decrease)")
    print(f"  ΔPR-AUC:  {diag_pr - screen_pr:.4f} ({100*(diag_pr - screen_pr)/diag_pr:.1f}% decrease)")

    if screen_auc > 0.75:
        print("\n✓ SCREENING MODEL PERFORMANCE: CLINICALLY ACCEPTABLE (AUC > 0.75)")
        print("  → Can be used for primary care risk stratification")
    elif screen_auc > 0.70:
        print("\n⚠ SCREENING MODEL PERFORMANCE: MODERATE (AUC 0.70-0.75)")
        print("  → May be useful with calibration or combined with other tests")
    else:
        print("\n✗ SCREENING MODEL PERFORMANCE: INSUFFICIENT (AUC < 0.70)")
        print("  → Need additional features or different modeling approach")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC curves
    ax1 = axes[0]
    colors = {'diagnostic': '#e74c3c', 'screening': '#2ecc71'}

    for model_type, result in results.items():
        # Aggregate ROC
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for fold_result in result['fold_results']:
            y_test = fold_result['y_test']
            y_proba = fold_result['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        auc_val = result['aggregated']['auc_roc']['mean']
        std_val = result['aggregated']['auc_roc']['std']

        label = f"{model_type.capitalize()}"
        if model_type == 'diagnostic':
            label += " (with CAPDbm)"
        else:
            label += " (without CAPDbm)"
        label += f"\nAUC = {auc_val:.3f} ± {std_val:.3f}"

        ax1.plot(mean_fpr, mean_tpr, color=colors[model_type], linewidth=3, label=label)

        # Confidence band
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(mean_fpr, tprs_lower, tprs_upper,
                        color=colors[model_type], alpha=0.2)

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Chance')
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_title('ROC Curve Comparison\nDiagnostic vs. Screening Models', fontweight='bold', fontsize=13)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Feature importance comparison
    ax2 = axes[1]

    # Get top features from screening model
    screen_importance = results['screening']['feature_importance']
    if screen_importance is not None:
        top_n = min(10, len(screen_importance))
        top_features = screen_importance.head(top_n)

        colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        bars = ax2.barh(range(len(top_features)), top_features['Mean_|SHAP|'],
                       color=colors_bar, edgecolor='black', linewidth=1.5)

        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['Feature'], fontweight='bold')
        ax2.set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=12)
        ax2.set_title('Screening Model - Top Predictors\n(Clinically Available Features)',
                     fontweight='bold', fontsize=13)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        # Add values
        for bar, val in zip(bars, top_features['Mean_|SHAP|']):
            ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/09_diagnostic_vs_screening_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: results/09_diagnostic_vs_screening_comparison.png")
    plt.close()

def generate_screening_recommendations(results):
    """Generate clinical recommendations for screening."""

    print("\n" + "="*80)
    print("CLINICAL RECOMMENDATIONS FOR FLD SCREENING")
    print("="*80)

    screen_importance = results['screening']['feature_importance']

    if screen_importance is not None:
        print("\nTOP 5 READILY-AVAILABLE PREDICTORS FOR PRIMARY CARE:\n")

        for idx, row in screen_importance.head(5).iterrows():
            print(f"  {idx+1}. {row['Feature']} (SHAP importance: {row['Mean_|SHAP|']:.3f})")

        print("\nPROPOSED SCREENING WORKFLOW:")
        print("""
        1. INITIAL ASSESSMENT (all patients):
           - Measure: BMI, waist circumference, blood pressure
           - Assess: Age, comorbidities (diabetes, hypertension)
           - Calculate: Risk score using ML model

        2. RISK STRATIFICATION:
           - High risk (predicted prob > 0.5): Refer for CAP/ultrasound
           - Moderate risk (0.3-0.5): Lifestyle counseling + reassess in 6 months
           - Low risk (< 0.3): Standard health maintenance

        3. INTERVENTION:
           - Weight loss (target: 7-10% body weight)
           - Increase physical activity
           - Dietary modification (reduce refined carbs, increase fiber)
           - Manage comorbidities (diabetes, hypertension)

        4. MONITORING:
           - Reassess risk score every 6-12 months
           - Track BMI and waist circumference changes
        """)

    # Performance adequacy
    screen_auc = results['screening']['aggregated']['auc_roc']['mean']
    screen_sens = results['screening']['aggregated']['recall']['mean']
    screen_spec = results['screening']['aggregated']['precision']['mean'] * screen_sens / \
                 (results['screening']['aggregated']['recall']['mean'] *
                  results['screening']['aggregated']['precision']['mean'] + 1e-10)

    print("\nMODEL CLINICAL PERFORMANCE:")
    print(f"  Sensitivity (recall): {screen_sens:.2%}")
    print(f"  Specificity:         ~{screen_spec:.2%} (estimated)")
    print(f"  AUC-ROC:             {screen_auc:.3f}")

    print("\nLIMITATIONS & FUTURE DIRECTIONS:")
    print("""
        - Current model based on cross-sectional data (need prospective validation)
        - Missing laboratory data (lipids, liver enzymes, glucose) could improve performance
        - Ethnicity-specific thresholds may be needed
        - External validation in different populations required
        - Consider ensemble with existing scores (FLI, HSI, NAFLD-LFS)
    """)

if __name__ == "__main__":

    # Train both models
    results = train_screening_models()

    # Compare
    compare_diagnostic_vs_screening(results)

    # Recommendations
    generate_screening_recommendations(results)

    # Save results
    with open('results/screening_vs_diagnostic_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*80)
    print("✓ Clinical screening analysis complete!")
    print("="*80)
