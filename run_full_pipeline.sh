#!/bin/bash

# FLD Classification: Complete ML Pipeline
# =========================================
# Executes full analysis from data exploration to clinical summary

echo "================================================================================"
echo "FLD ML PIPELINE - COMPLETE EXECUTION"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p results

# Step 1: Data Exploration
echo "STEP 1/6: Data Exploration"
echo "--------------------------------------------------------------------------------"
python 01_data_exploration.py
if [ $? -ne 0 ]; then
    echo "ERROR: Data exploration failed"
    exit 1
fi
echo ""

# Step 2: Nested CV Modeling (with CAPDbm - diagnostic model)
echo "STEP 2/6: Nested CV Modeling (RandomForest, XGBoost, LightGBM)"
echo "--------------------------------------------------------------------------------"
python 02_nested_cv_modeling.py
if [ $? -ne 0 ]; then
    echo "ERROR: Nested CV modeling failed"
    exit 1
fi
echo ""

# Step 3: Visualizations
echo "STEP 3/6: Generate Visualizations (ROC, PR, Calibration, etc.)"
echo "--------------------------------------------------------------------------------"
python 03_visualize_results.py
if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed"
    exit 1
fi
echo ""

# Step 4: SHAP Analysis
echo "STEP 4/6: SHAP Feature Importance Analysis"
echo "--------------------------------------------------------------------------------"
python 04_shap_analysis.py
if [ $? -ne 0 ]; then
    echo "ERROR: SHAP analysis failed"
    exit 1
fi
echo ""

# Step 5: Clinical Screening Model (without CAPDbm)
echo "STEP 5/6: Clinical Screening Model (Excluding CAPDbm)"
echo "--------------------------------------------------------------------------------"
python 05_clinical_screening_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Screening model training failed"
    exit 1
fi
echo ""

# Step 6: Clinical Summary
echo "STEP 6/6: Generate Clinical Summary"
echo "--------------------------------------------------------------------------------"
python 06_clinical_summary.py
if [ $? -ne 0 ]; then
    echo "ERROR: Summary generation failed"
    exit 1
fi
echo ""

# Success
echo "================================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated outputs:"
echo "  - results/01_exploratory_analysis.png"
echo "  - results/02_roc_curves.png"
echo "  - results/03_pr_curves.png"
echo "  - results/04_model_comparison.png"
echo "  - results/05_calibration_curves.png"
echo "  - results/06_fold_variability.png"
echo "  - results/07_shap_*.png (per model)"
echo "  - results/08_feature_importance_comparison.png"
echo "  - results/09_diagnostic_vs_screening_comparison.png"
echo "  - results/metrics_summary.csv"
echo "  - results/feature_importance_*.csv (per model)"
echo "  - results/nested_cv_results.pkl"
echo "  - results/screening_vs_diagnostic_results.pkl"
echo "  - results/CLINICAL_SUMMARY.txt"
echo ""
echo "✓ Ready for manuscript preparation and clinical deployment!"
