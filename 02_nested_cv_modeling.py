"""
FLD Classification: Rigorous Nested Cross-Validation Pipeline
==============================================================
Author: Medical ML Research Assistant
Purpose: Leakage-free nested CV with RandomForest, XGBoost, LightGBM
         - Outer loop: 5-fold CV for performance estimation
         - Inner loop: 3-fold CV for hyperparameter tuning
         - SHAP-based interpretability
         - Comprehensive metrics with confidence intervals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import pickle
import json
from datetime import datetime

# Sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

# Gradient boosting
import xgboost as xgb
import lightgbm as lgb

# Interpretability
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_path': 'non_invasive_clean.csv',
    'target': 'fld',
    'exclude_features': ['id', 'fld'],  # CAPDbm included as predictor

    # Cross-validation strategy
    'outer_folds': 5,
    'inner_folds': 3,
    'random_state': 42,

    # Models to train
    'models': ['RandomForest', 'XGBoost', 'LightGBM'],

    # Computational
    'n_jobs': -1,
    'verbose': 1,

    # Output
    'output_dir': 'results',
    'save_models': True,
}

# =============================================================================
# HYPERPARAMETER GRIDS
# =============================================================================

def get_hyperparameter_grids(scale_pos_weight):
    """
    Define hyperparameter search spaces.

    Strategy:
    - Prioritize generalization over training performance
    - Include regularization parameters
    - Handle class imbalance explicitly
    """

    grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5],
            'class_weight': ['balanced', 'balanced_subsample'],
            'bootstrap': [True],
            'random_state': [42]
        },

        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.5],  # Regularization
            'reg_alpha': [0, 0.1, 1],  # L1 regularization
            'reg_lambda': [1, 5, 10],  # L2 regularization
            'scale_pos_weight': [scale_pos_weight],  # Handle imbalance
            'random_state': [42]
        },

        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10],
            'scale_pos_weight': [scale_pos_weight],
            'random_state': [42],
            'verbose': [-1]
        }
    }

    return grids

# =============================================================================
# NESTED CROSS-VALIDATION ENGINE
# =============================================================================

class NestedCVPipeline:
    """
    Rigorous nested cross-validation with leakage prevention.

    Key features:
    - Independent preprocessing for each fold
    - Hyperparameter tuning in inner loop
    - Performance estimation in outer loop
    - SHAP values computed per fold
    - Comprehensive metrics tracking
    """

    def __init__(self, config):
        self.config = config
        self.results = {
            'fold_results': [],
            'aggregated_metrics': {},
            'shap_values': {},
            'best_params': {},
            'predictions': []
        }

    def load_data(self):
        """Load and prepare data."""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        df = pd.read_csv(self.config['data_path'])
        print(f"✓ Loaded {df.shape[0]} samples")

        # Separate features and target
        feature_cols = [col for col in df.columns
                       if col not in self.config['exclude_features']]

        X = df[feature_cols].values
        y = df[self.config['target']].values
        feature_names = feature_cols

        # Calculate class imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / n_pos

        print(f"✓ Features: {len(feature_names)}")
        print(f"✓ Class distribution: {n_neg} negative, {n_pos} positive")
        print(f"✓ Scale pos weight: {scale_pos_weight:.3f}")

        return X, y, feature_names, scale_pos_weight

    def get_model(self, model_name):
        """Initialize model based on name."""
        if model_name == 'RandomForest':
            return RandomForestClassifier()
        elif model_name == 'XGBoost':
            return xgb.XGBClassifier(eval_metric='logloss')
        elif model_name == 'LightGBM':
            return lgb.LGBMClassifier()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def compute_metrics(self, y_true, y_pred, y_proba):
        """Compute comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba),
        }

        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]

        # Specificity
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

        return metrics

    def train_model_nested_cv(self, model_name, X, y, feature_names, scale_pos_weight):
        """
        Train single model with rigorous nested CV.

        CRITICAL: Preprocessing fitted independently in each fold!
        """
        print("\n" + "="*80)
        print(f"TRAINING: {model_name}")
        print("="*80)

        # Get hyperparameter grid
        param_grids = get_hyperparameter_grids(scale_pos_weight)
        param_grid = param_grids[model_name]

        # Outer CV for performance estimation
        outer_cv = StratifiedKFold(
            n_splits=self.config['outer_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )

        fold_results = []
        fold_shap_values = []
        all_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            print(f"\n--- Outer Fold {fold_idx}/{self.config['outer_folds']} ---")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # =====================================================================
            # CRITICAL: FIT SCALER ONLY ON TRAINING DATA (NO LEAKAGE!)
            # =====================================================================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # Use same scaler

            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedKFold(
                n_splits=self.config['inner_folds'],
                shuffle=True,
                random_state=self.config['random_state']
            )

            # Initialize model
            model = self.get_model(model_name)

            # Grid search with inner CV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',  # Primary metric for tuning
                n_jobs=self.config['n_jobs'],
                verbose=0,
                refit=True
            )

            # Fit on training data
            grid_search.fit(X_train_scaled, y_train)

            # Best model from inner CV
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            print(f"✓ Best inner CV AUC: {grid_search.best_score_:.4f}")
            print(f"✓ Best params: {best_params}")

            # Predict on held-out test fold
            y_pred = best_model.predict(X_test_scaled)
            y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

            # Compute metrics
            metrics = self.compute_metrics(y_test, y_pred, y_proba)

            print(f"✓ Test AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"✓ Test PR-AUC:  {metrics['auc_pr']:.4f}")
            print(f"✓ Test F1:      {metrics['f1']:.4f}")

            # Store fold results
            fold_results.append({
                'fold': fold_idx,
                'metrics': metrics,
                'best_params': best_params,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            })

            # Store predictions for later analysis
            all_predictions.extend([{
                'fold': fold_idx,
                'true_label': int(yt),
                'predicted_label': int(yp),
                'predicted_proba': float(ypr)
            } for yt, yp, ypr in zip(y_test, y_pred, y_proba)])

            # =====================================================================
            # SHAP VALUES (for interpretability)
            # =====================================================================
            try:
                print("  Computing SHAP values...")

                # Use TreeExplainer for tree-based models
                if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test_scaled)

                    # For binary classification, some models return list
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Positive class

                    fold_shap_values.append({
                        'fold': fold_idx,
                        'shap_values': shap_values,
                        'X_test': X_test_scaled,
                        'feature_names': feature_names
                    })

                    print("  ✓ SHAP values computed")
            except Exception as e:
                print(f"  ⚠ SHAP computation failed: {e}")

        # =====================================================================
        # AGGREGATE METRICS ACROSS FOLDS
        # =====================================================================
        print("\n" + "-"*80)
        print("AGGREGATED RESULTS")
        print("-"*80)

        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc_roc', 'auc_pr']
        aggregated = {}

        for metric in metric_names:
            values = [fold['metrics'][metric] for fold in fold_results]
            mean = np.mean(values)
            std = np.std(values)
            ci_95 = 1.96 * std / np.sqrt(len(values))

            aggregated[metric] = {
                'mean': mean,
                'std': std,
                'ci_95': ci_95,
                'values': values
            }

            print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f} (95% CI: [{mean-ci_95:.4f}, {mean+ci_95:.4f}])")

        # Check for suspicious perfect scores
        if aggregated['auc_roc']['mean'] > 0.99:
            print("\n⚠⚠⚠ RED FLAG: Near-perfect AUC-ROC detected!")
            print("    Possible causes:")
            print("    - Data leakage (check preprocessing)")
            print("    - Target leakage (e.g., CAPDbm is diagnostic, not truly predictive)")
            print("    - Overfitting (check if train >> test)")

        return {
            'model_name': model_name,
            'fold_results': fold_results,
            'aggregated_metrics': aggregated,
            'shap_values': fold_shap_values,
            'predictions': all_predictions
        }

    def run(self):
        """Execute full nested CV pipeline."""

        print("\n" + "="*80)
        print("FATTY LIVER DISEASE (FLD) - NESTED CV MODELING")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        X, y, feature_names, scale_pos_weight = self.load_data()

        # Store data info
        self.results['data_info'] = {
            'n_samples': len(y),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'class_distribution': {
                'negative': int((y == 0).sum()),
                'positive': int((y == 1).sum())
            },
            'scale_pos_weight': float(scale_pos_weight)
        }

        # Train each model
        for model_name in self.config['models']:
            model_results = self.train_model_nested_cv(
                model_name, X, y, feature_names, scale_pos_weight
            )
            self.results['fold_results'].append(model_results)

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)

        return self.results

    def save_results(self, results):
        """Save results to disk."""
        import os

        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Save full results as pickle
        with open(f'{output_dir}/nested_cv_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to {output_dir}/")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Initialize pipeline
    pipeline = NestedCVPipeline(CONFIG)

    # Run nested CV
    results = pipeline.run()

    # Save results
    pipeline.save_results(results)

    print("\n✓ Nested CV modeling complete!")
    print("\nNext steps:")
    print("  → Generate comprehensive visualizations (script 03)")
    print("  → Analyze SHAP feature importances (script 04)")
    print("  → Write clinical interpretation summary (script 05)")
