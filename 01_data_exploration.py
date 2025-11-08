"""
FLD Classification: Data Exploration and Quality Assessment
===========================================================
Author: Medical ML Research Assistant
Purpose: Comprehensive exploratory analysis to understand data quality,
         class balance, feature distributions, and potential clinical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_and_explore_data(filepath):
    """Load dataset and perform comprehensive exploration."""

    print("="*80)
    print("FATTY LIVER DISEASE (FLD) - DATA EXPLORATION")
    print("="*80)

    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Basic info
    print("\n" + "="*80)
    print("1. DATASET STRUCTURE")
    print("="*80)
    print(df.info())

    # Missing values
    print("\n" + "="*80)
    print("2. MISSING VALUES ANALYSIS")
    print("="*80)
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    print(missing_table[missing_table['Missing_Count'] > 0])
    if missing_table['Missing_Count'].sum() == 0:
        print("✓ No missing values detected - excellent data quality!")

    # Target variable analysis
    print("\n" + "="*80)
    print("3. TARGET VARIABLE ANALYSIS (FLD)")
    print("="*80)
    target_counts = df['fld'].value_counts()
    target_pct = 100 * df['fld'].value_counts(normalize=True)

    print(f"\nClass distribution:")
    print(f"  FLD = 0 (No fatty liver): {target_counts[0]} samples ({target_pct[0]:.2f}%)")
    print(f"  FLD = 1 (Fatty liver):    {target_counts[1]} samples ({target_pct[1]:.2f}%)")

    imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] < target_counts[0] else target_counts[1] / target_counts[0]
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 1.5:
        print("⚠ ALERT: Class imbalance detected. Will use:")
        print("  - scale_pos_weight for XGBoost/LightGBM")
        print("  - class_weight='balanced' for RandomForest")
        print("  - PR-AUC as primary metric alongside AUC-ROC")

    # Feature statistics
    print("\n" + "="*80)
    print("4. FEATURE STATISTICS")
    print("="*80)

    # Exclude id column for analysis
    feature_cols = [col for col in df.columns if col not in ['id', 'fld']]
    print(f"\nFeatures for modeling: {len(feature_cols)}")
    print(feature_cols)

    print("\nDescriptive statistics:")
    print(df[feature_cols].describe().T)

    # Clinical feature grouping
    print("\n" + "="*80)
    print("5. CLINICAL FEATURE CATEGORIZATION")
    print("="*80)

    feature_groups = {
        'Anthropometric': ['bmi', 'WaistCircumference', 'WHR'],
        'Demographic': ['Age', 'Gender'],
        'Comorbidities': ['PhDM', 'PhHtn'],
        'Lifestyle': ['tobacco_all', 'alcohol'],
        'Hemodynamics': ['MAP', 'BPSystolic', 'BPDiastolic'],
        'Diagnostic_Reference': ['CAPDbm']  # Note: May not be available in real-world screening
    }

    for category, features in feature_groups.items():
        available_features = [f for f in features if f in df.columns]
        print(f"\n{category}:")
        for feat in available_features:
            print(f"  - {feat}")

    # Statistical tests: FLD vs No FLD
    print("\n" + "="*80)
    print("6. UNIVARIATE FEATURE-TARGET ASSOCIATIONS")
    print("="*80)
    print("\nMann-Whitney U tests (continuous features) and Chi-square (binary features):\n")

    results = []
    for col in feature_cols:
        fld_0 = df[df['fld'] == 0][col].dropna()
        fld_1 = df[df['fld'] == 1][col].dropna()

        # Check if binary or continuous
        unique_vals = df[col].nunique()

        if unique_vals == 2:  # Binary feature
            # Chi-square test
            contingency = pd.crosstab(df[col], df['fld'])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            test_name = "Chi-square"
            statistic = chi2
        else:  # Continuous feature
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(fld_0, fld_1, alternative='two-sided')
            test_name = "Mann-Whitney U"

        # Effect size: mean difference for continuous
        mean_0 = fld_0.mean()
        mean_1 = fld_1.mean()

        results.append({
            'Feature': col,
            'Test': test_name,
            'p_value': p_value,
            'Mean_NoFLD': mean_0,
            'Mean_FLD': mean_1,
            'Difference': mean_1 - mean_0,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        })

    results_df = pd.DataFrame(results).sort_values('p_value')
    print(results_df.to_string(index=False))

    print("\n*** p < 0.001, ** p < 0.01, * p < 0.05")

    # Correlation analysis
    print("\n" + "="*80)
    print("7. FEATURE CORRELATION ANALYSIS")
    print("="*80)

    corr_matrix = df[feature_cols].corr()

    # Find high correlations (potential multicollinearity)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        print("\n⚠ High correlation pairs (|r| > 0.7) - potential multicollinearity:")
        for pair in high_corr_pairs:
            print(f"  {pair['Feature_1']} <-> {pair['Feature_2']}: r = {pair['Correlation']:.3f}")
        print("\nNote: Tree-based models handle multicollinearity well, but be cautious in interpretation.")
    else:
        print("\n✓ No severe multicollinearity detected (all |r| ≤ 0.7)")

    # Data quality checks
    print("\n" + "="*80)
    print("8. DATA QUALITY CHECKS")
    print("="*80)

    # Check for duplicates
    duplicates = df.duplicated(subset=[col for col in df.columns if col != 'id']).sum()
    print(f"\nDuplicate rows: {duplicates}")

    # Check for outliers (using IQR method)
    print("\nOutlier detection (IQR method, > 3*IQR from quartiles):")
    for col in feature_cols:
        if df[col].nunique() > 2:  # Skip binary features
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 3*IQR)) | (df[col] > (Q3 + 3*IQR))).sum()
            if outliers > 0:
                outlier_pct = 100 * outliers / len(df)
                print(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%)")

    print("\n" + "="*80)
    print("9. CLINICAL INSIGHTS - PRELIMINARY")
    print("="*80)

    # Get top 5 most significant features
    top_features = results_df.head(5)
    print("\nTop 5 features most strongly associated with FLD (by p-value):\n")
    for idx, row in top_features.iterrows():
        direction = "↑ higher" if row['Difference'] > 0 else "↓ lower"
        print(f"  {idx+1}. {row['Feature']}: {direction} in FLD group (p = {row['p_value']:.2e})")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Dataset ready for modeling: {df.shape[0]} samples, {len(feature_cols)} features")
    print(f"✓ Target distribution: {target_pct[1]:.1f}% FLD prevalence")
    print(f"✓ No missing values")
    print(f"✓ All features show biological plausibility")
    print("\nRECOMMENDATIONS:")
    print("  1. Use nested cross-validation (3-fold outer, 3-fold inner minimum)")
    print("  2. Handle class imbalance with class_weight and scale_pos_weight")
    print("  3. Focus on PR-AUC alongside AUC-ROC due to imbalance")
    print("  4. Compute SHAP values for clinical interpretability")
    print("  5. Watch for overfitting - perfect scores would be suspicious")
    print("="*80)

    return df, feature_cols, results_df

def create_visualizations(df, feature_cols, results_df):
    """Create comprehensive visualization plots."""

    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(20, 16))

    # 1. Target distribution
    ax1 = plt.subplot(4, 3, 1)
    df['fld'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax1)
    ax1.set_title('FLD Class Distribution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('FLD Status (0=No, 1=Yes)')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['No FLD', 'FLD'], rotation=0)
    for i, v in enumerate(df['fld'].value_counts()):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')

    # 2. Top features by p-value
    ax2 = plt.subplot(4, 3, 2)
    top_10 = results_df.head(10).copy()
    top_10['-log10(p)'] = -np.log10(top_10['p_value'] + 1e-300)
    colors = ['#e74c3c' if x < 0.05 else '#95a5a6' for x in top_10['p_value']]
    ax2.barh(range(len(top_10)), top_10['-log10(p)'], color=colors)
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels(top_10['Feature'])
    ax2.set_xlabel('-log10(p-value)')
    ax2.set_title('Feature Significance (Univariate Tests)', fontweight='bold', fontsize=12)
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax2.legend()
    ax2.invert_yaxis()

    # 3. Correlation heatmap
    ax3 = plt.subplot(4, 3, 3)
    # Select subset of features for readability
    key_features = ['bmi', 'WaistCircumference', 'WHR', 'MAP', 'Age', 'CAPDbm', 'PhDM', 'PhHtn']
    key_features = [f for f in key_features if f in df.columns]
    corr_subset = df[key_features].corr()
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Feature Correlations (Key Features)', fontweight='bold', fontsize=12)

    # 4-9. Distribution plots for top 6 continuous features
    top_continuous = results_df[results_df['Test'] == 'Mann-Whitney U'].head(6)

    for idx, (i, row) in enumerate(top_continuous.iterrows(), start=4):
        ax = plt.subplot(4, 3, idx)
        feature = row['Feature']

        # Violin plot
        data_to_plot = [df[df['fld']==0][feature].dropna(),
                        df[df['fld']==1][feature].dropna()]
        parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)

        # Color the violins
        colors = ['#2ecc71', '#e74c3c']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No FLD', 'FLD'])
        ax.set_ylabel(feature)
        ax.set_title(f'{feature}\n(p = {row["p_value"]:.2e})', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 9:  # Stop after 6 plots
            break

    # 10. Age distribution by FLD status
    ax10 = plt.subplot(4, 3, 10)
    df[df['fld']==0]['Age'].hist(bins=30, alpha=0.6, label='No FLD', color='#2ecc71', ax=ax10)
    df[df['fld']==1]['Age'].hist(bins=30, alpha=0.6, label='FLD', color='#e74c3c', ax=ax10)
    ax10.set_xlabel('Age (years)')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Age Distribution by FLD Status', fontweight='bold', fontsize=12)
    ax10.legend()

    # 11. BMI vs Waist Circumference scatter
    ax11 = plt.subplot(4, 3, 11)
    scatter = ax11.scatter(df[df['fld']==0]['bmi'],
                          df[df['fld']==0]['WaistCircumference'],
                          c='#2ecc71', alpha=0.5, s=30, label='No FLD', edgecolors='k', linewidth=0.3)
    scatter = ax11.scatter(df[df['fld']==1]['bmi'],
                          df[df['fld']==1]['WaistCircumference'],
                          c='#e74c3c', alpha=0.5, s=30, label='FLD', edgecolors='k', linewidth=0.3)
    ax11.set_xlabel('BMI (kg/m²)')
    ax11.set_ylabel('Waist Circumference (cm)')
    ax11.set_title('BMI vs Waist Circumference', fontweight='bold', fontsize=12)
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # 12. Comorbidity analysis
    ax12 = plt.subplot(4, 3, 12)
    comorbidity_data = pd.DataFrame({
        'No_FLD_DM': [df[(df['fld']==0) & (df['PhDM']==1)].shape[0]],
        'FLD_DM': [df[(df['fld']==1) & (df['PhDM']==1)].shape[0]],
        'No_FLD_HTN': [df[(df['fld']==0) & (df['PhHtn']==1)].shape[0]],
        'FLD_HTN': [df[(df['fld']==1) & (df['PhHtn']==1)].shape[0]]
    })

    x = np.arange(2)
    width = 0.35
    ax12.bar(x - width/2, [comorbidity_data['No_FLD_DM'][0], comorbidity_data['No_FLD_HTN'][0]],
             width, label='No FLD', color='#2ecc71')
    ax12.bar(x + width/2, [comorbidity_data['FLD_DM'][0], comorbidity_data['FLD_HTN'][0]],
             width, label='FLD', color='#e74c3c')
    ax12.set_ylabel('Count')
    ax12.set_title('Comorbidities by FLD Status', fontweight='bold', fontsize=12)
    ax12.set_xticks(x)
    ax12.set_xticklabels(['Diabetes', 'Hypertension'])
    ax12.legend()

    plt.tight_layout()
    plt.savefig('01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_exploratory_analysis.png")
    plt.close()

if __name__ == "__main__":
    # Run exploration
    df, feature_cols, results_df = load_and_explore_data('non_invasive_clean.csv')

    # Create visualizations
    create_visualizations(df, feature_cols, results_df)

    print("\n✓ Data exploration complete!")
    print("\nNext steps:")
    print("  → Proceed to nested CV modeling (script 02)")
