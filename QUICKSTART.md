# FLD ML Pipeline - Quick Start Guide

## 🚀 Running the Complete Pipeline

### Option 1: Automated Full Pipeline (Recommended)

```bash
bash run_full_pipeline.sh
```

**Execution time:** ~30-60 minutes (depending on hardware)
**Output:** All figures, tables, and summary report in `results/` directory

---

### Option 2: Step-by-Step Execution

```bash
# 1. Data Exploration (2-3 minutes)
python 01_data_exploration.py
# → Creates: 01_exploratory_analysis.png

# 2. Nested CV Training (20-40 minutes - MOST TIME-CONSUMING)
python 02_nested_cv_modeling.py
# → Creates: results/nested_cv_results.pkl
# → Trains RandomForest, XGBoost, LightGBM with 5-fold nested CV

# 3. Visualization Generation (1-2 minutes)
python 03_visualize_results.py
# → Creates: ROC curves, PR curves, calibration plots, etc.

# 4. SHAP Analysis (2-3 minutes)
python 04_shap_analysis.py
# → Creates: SHAP importance plots, feature comparison

# 5. Screening Model (10-15 minutes)
python 05_clinical_screening_model.py
# → Creates: Clinically-realistic model without CAPDbm

# 6. Clinical Summary (< 1 minute)
python 06_clinical_summary.py
# → Creates: CLINICAL_SUMMARY.txt with executive summary
```

---

## 📊 Key Outputs

### Figures (publication-ready, 300 DPI)

| File | Description |
|------|-------------|
| `01_exploratory_analysis.png` | Dataset overview, class distribution, feature significance |
| `02_roc_curves.png` | ROC curves for all models with confidence bands |
| `03_pr_curves.png` | Precision-Recall curves (critical for imbalanced data) |
| `04_model_comparison.png` | Side-by-side metrics comparison |
| `05_calibration_curves.png` | Probability calibration assessment |
| `06_fold_variability.png` | Cross-fold stability analysis |
| `07_shap_*.png` | SHAP feature importance (per model) |
| `08_feature_importance_comparison.png` | Consensus features across models |
| `09_diagnostic_vs_screening_comparison.png` | Performance with/without CAPDbm |

### Tables (CSV format)

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Comprehensive metrics with 95% CIs |
| `feature_importance_*.csv` | SHAP values per model |

### Reports

| File | Description |
|------|-------------|
| `CLINICAL_SUMMARY.txt` | Executive summary with clinical recommendations |

---

## 🔍 Understanding the Results

### Expected Performance

**Diagnostic Model (with CAPDbm):**
- AUC-ROC: **~1.0000** (perfect - indicates CAPDbm is diagnostic, not predictive)
- This validates pipeline integrity but has limited clinical utility

**Screening Model (without CAPDbm):**
- AUC-ROC: **0.75-0.85** (clinically acceptable for risk stratification)
- Uses only: BMI, waist circumference, blood pressure, age, comorbidities
- **This is the clinically-useful model for primary care**

### Top Predictive Features (Screening Model)

1. **BMI** - Obesity drives hepatic lipogenesis
2. **Waist Circumference** - Central adiposity marker
3. **Blood Pressure (MAP/Systolic)** - Metabolic syndrome component
4. **Age** - Cumulative metabolic stress
5. **Diabetes Status** - Shared insulin resistance pathway

---

## 🚨 Red Flags & Validation

### Built-in Quality Checks

The pipeline automatically detects and reports:

✓ **Perfect AUC detection** - Flags potential target leakage
✓ **Cross-fold variability** - Assesses model stability
✓ **Calibration assessment** - Validates probability reliability
✓ **Data leakage prevention** - Independent preprocessing per fold

### Methodological Rigor

**Nested Cross-Validation:**
- Outer loop (5-fold): Performance estimation
- Inner loop (3-fold): Hyperparameter tuning
- **Prevents optimistic bias** from parameter selection

**Class Imbalance Handling:**
- `scale_pos_weight` for XGBoost/LightGBM
- `class_weight='balanced'` for RandomForest
- **PR-AUC** as primary metric alongside AUC-ROC

---

## 💡 Clinical Translation

### Proposed Screening Workflow

```
PRIMARY CARE VISIT
    ↓
Measure: BMI, waist circumference, blood pressure
Assess: Age, diabetes, hypertension
    ↓
Calculate ML Risk Score
    ↓
├─ High Risk (>0.5) → Refer for CAP/Ultrasound
├─ Moderate Risk (0.3-0.5) → Lifestyle counseling, reassess 6mo
└─ Low Risk (<0.3) → Standard health maintenance
```

### Implementation Requirements

1. **Validation:** External cohort validation
2. **Interface:** Web calculator or EHR integration
3. **Monitoring:** Prospective performance tracking
4. **Ethics:** Algorithmic bias assessment

---

## 📚 Interpreting SHAP Values

**SHAP (SHapley Additive exPlanations)** quantifies each feature's contribution to predictions.

**High SHAP value → Feature strongly pushes prediction toward FLD**

Example interpretation:
- **BMI = 30 kg/m²**: SHAP = +0.15 → Increases FLD probability
- **Age = 55 years**: SHAP = +0.08 → Modest increase
- **Waist = 75 cm**: SHAP = -0.12 → Decreases FLD probability

**Clinical Insight:** Anthropometric measures (BMI, waist) consistently show highest SHAP values, validating visceral adiposity as central driver of NAFLD.

---

## ⚠️ Limitations

**Current pipeline limitations:**

1. **Cross-sectional design** - Cannot establish causation
2. **Missing lab data** - Lipids, ALT, glucose would improve performance
3. **Single-center** - Needs multi-site validation
4. **Ethnicity** - May need population-specific calibration
5. **Severity staging** - Does not predict NASH or fibrosis progression

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** "ModuleNotFoundError: No module named 'pandas'"
**Solution:** `pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm shap`

**Issue:** Grid search taking too long (>2 hours)
**Solution:** Reduce hyperparameter grid size in `02_nested_cv_modeling.py`

**Issue:** SHAP computation fails
**Solution:** Non-critical error; SHAP plots may be incomplete but models still valid

**Issue:** Perfect AUC (1.0000) detected
**Solution:** This is EXPECTED when CAPDbm is included - see screening model for clinical utility

---

## 📖 Next Steps

1. **Review results:** Check `CLINICAL_SUMMARY.txt` for executive summary
2. **Validate findings:** Compare with existing NAFLD risk scores (FLI, HSI)
3. **External validation:** Test on independent dataset
4. **Refinement:** Add laboratory predictors if available
5. **Deployment:** Build web interface or EHR integration

---

## 📞 Support

**Technical questions:** Review script comments and README.md
**Clinical interpretation:** See CLINICAL_SUMMARY.txt
**Methodology details:** Check inline documentation in Python scripts

---

**Version:** 1.0
**Last Updated:** 2025-11-08
**Status:** ✓ Production-ready for research validation
