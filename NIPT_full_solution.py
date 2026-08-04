#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT Mathematical Modeling — Complete Solving Script
=====================================================
2025 MCM/ICM Problem C: NIPT Timing Selection & Fetal Abnormality Detection

Modules:
  1. Data Loading
  2. Data Preprocessing
  3. Q1 — Y Concentration vs Age + BMI Relationship Model
  4. Q2 — BMI-Based Grouping → Optimal NIPT Timing
  5. Q3 — Multi-Factor Grouping → Optimal NIPT Timing
  6. Q4 — Female Fetus Abnormality Classification
  7. Results Aggregation & Output

Author: MathModeling-skills Workflow
Date:   2025-08-03
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   LogisticRegression)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              silhouette_score, classification_report)
from sklearn.svm import SVC
from pathlib import Path

# ============================================================================
# Optional statsmodels import (for mixed linear model; OLS uses sklearn+scipy)
# ============================================================================
_STATSMODELS_AVAILABLE = False
try:
    import statsmodels.api as _sm
    from statsmodels.formula.api import mixedlm as _mixedlm
    _STATSMODELS_AVAILABLE = True
except ImportError:
    pass  # statsmodels not installed; mixed-model step will skip gracefully


# ============================================================================
# Helper: OLS diagnostics via sklearn (replaces statsmodels OLS)
# ============================================================================
class _OLSResult:
    """Minimal statsmodels-like result container computed from sklearn."""
    def __init__(self, coef_names, beta, se, tstat, pvalues, rsquared, rsquared_adj, n, df_resid):
        self.params = dict(zip(coef_names, beta))
        self.bse = dict(zip(coef_names, se))
        self.tvalues = dict(zip(coef_names, tstat))
        self.pvalues = dict(zip(coef_names, pvalues))
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj
        self.nobs = n
        self.df_resid = df_resid

    def predict(self, X_with_const):
        """X_with_const must include intercept column."""
        beta = np.array([self.params[n] for n in self.params.keys()])
        return X_with_const @ beta


def _ols_fit(y, X, feature_names):
    """
    Fit OLS using sklearn + manual diagnostics.
    Returns an _OLSResult with .params, .pvalues, .rsquared, .rsquared_adj, .predict().

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p) — does NOT include intercept (added internally)
    feature_names : list of str
    """
    from sklearn.linear_model import LinearRegression as _LR

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, p = X_arr.shape

    # Add intercept
    Xc = np.column_stack([np.ones(n), X_arr])
    names = ['const'] + list(feature_names)

    # Fit via sklearn
    lr = _LR(fit_intercept=False).fit(Xc, y_arr)
    y_pred = lr.predict(Xc)
    resid = y_arr - y_pred

    # R² and Adj R²
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1)

    # Standard errors, t-stats, p-values
    mse = ss_res / max(1, n - p - 1)
    try:
        XtX_inv = np.linalg.inv(Xc.T @ Xc)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 1e-30))
    tstat = lr.coef_ / se
    from scipy.stats import t as _tdist
    pvalues = 2.0 * (1.0 - _tdist.cdf(np.abs(tstat), df=max(1, n - p - 1)))

    return _OLSResult(names, lr.coef_, se, tstat, pvalues, r2, r2_adj, n, max(1, n - p - 1))

# ============================================================================
# Global Configuration
# ============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = Path(r'C:\Users\imagination\Desktop\NIPT-MathModel\workspace\data_raw')
OUTPUT_DIR = Path(r'C:\Users\imagination\Desktop\NIPT-MathModel\results')
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'

for d in [OUTPUT_DIR, FIGURE_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
Y_THRESHOLD = 0.04       # Y concentration ≥ 4% threshold
GC_LOW, GC_HIGH = 0.40, 0.60  # Normal GC range
Z_THRESHOLD = 3.0        # |Z| > 3 = potential aneuploidy
EARLY_WEEK = 12          # ≤12 weeks: low risk
MID_WEEK = 27            # 13–27: high risk
LATE_WEEK = 28           # 28+: very high risk


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_data():
    """Load and merge male/female NIPT data from the attachment Excel file."""
    print("=" * 60)
    print("1. DATA LOADING")
    print("=" * 60)

    excel_path = DATA_DIR / '附件.xlsx'

    df_male = pd.read_excel(excel_path, sheet_name='男胎检测数据')
    df_female = pd.read_excel(excel_path, sheet_name='女胎检测数据')

    print(f"  Male data:   {df_male.shape[0]} rows × {df_male.shape[1]} cols")
    print(f"  Female data: {df_female.shape[0]} rows × {df_female.shape[1]} cols")

    return df_male, df_female


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
def parse_gestational_weeks(week_str):
    """Parse gestational week string like '11w+6' → float weeks."""
    if pd.isna(week_str):
        return np.nan
    week_str = str(week_str).strip()
    if 'w' in week_str:
        parts = week_str.replace('+', 'w').split('w')
        weeks = float(parts[0])
        days = float(parts[1]) if len(parts) > 1 and parts[1] else 0
        return weeks + days / 7.0
    return np.nan


def preprocess_data(df_male, df_female):
    """Clean and preprocess both datasets."""
    print("\n" + "=" * 60)
    print("2. DATA PREPROCESSING")
    print("=" * 60)

    for df, name in [(df_male, 'Male'), (df_female, 'Female')]:
        # Standardize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Rename key columns for easier access
        col_map = {
            '序号': 'sample_id',
            '孕妇代码': 'mother_id',
            '年龄': 'age',
            '身高': 'height_cm',
            '体重': 'weight_kg',
            '末次月经': 'lmp_date',
            'IVF妊娠': 'ivf_type',
            '检测日期': 'test_date',
            '检测抽血次数': 'draw_count',
            '检测孕周': 'gestational_weeks_raw',
            '孕妇BMI': 'bmi',
            '原始读段数': 'raw_reads',
            '在参考基因组上比对的比例': 'align_ratio',
            '重复读段的比例': 'dup_ratio',
            '唯一比对的读段数': 'unique_reads',
            'GC含量': 'gc_content',
            '13号染色体的Z值': 'z_chr13',
            '18号染色体的Z值': 'z_chr18',
            '21号染色体的Z值': 'z_chr21',
            'X染色体的Z值': 'z_chrX',
            'Y染色体的Z值': 'z_chrY',
            'Y染色体浓度': 'y_conc',
            'X染色体浓度': 'x_conc',
            '13号染色体的GC含量': 'gc_chr13',
            '18号染色体的GC含量': 'gc_chr18',
            '21号染色体的GC含量': 'gc_chr21',
            '被过滤掉读段数的比例': 'filtered_ratio',
            '染色体的非整倍体': 'aneuploidy',
            '怀孕次数': 'gravidity',
            '生产次数': 'parity',
            '胎儿是否健康': 'is_healthy',
        }
        df.rename(columns=col_map, inplace=True)

        # Parse gestational weeks
        df['gestational_weeks'] = df['gestational_weeks_raw'].apply(
            parse_gestational_weeks)

        # Convert dates
        for dc in ['lmp_date', 'test_date']:
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors='coerce')

        # --- Missing value handling ---
        n_before = df.isnull().sum().sum()

        # BMI: interpolate by mother (linear interpolation across visits)
        if 'bmi' in df.columns:
            df['bmi'] = df.groupby('mother_id')['bmi'].transform(
                lambda x: x.interpolate(method='linear').bfill().ffill())

        # y_conc / z_chrY: fill with group median (only for male data)
        for col in ['y_conc', 'z_chrY']:
            if col in df.columns and df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Categorical: fill mode (skip 'aneuploidy' — None means Normal)
        for col in ['ivf_type', 'is_healthy', 'lmp_date']:
            if col in df.columns:
                if col == 'lmp_date':
                    df[col] = df.groupby('mother_id')[col].transform(
                        lambda x: x.bfill().ffill())
                else:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])

        # Remaining numeric: median by mother group
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df.groupby('mother_id')[col].transform(
                    lambda x: x.fillna(x.median()))

        # Final fallback: fill remaining with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        n_after = df.isnull().sum().sum()
        print(f"  {name}: {n_before} → {n_after} missing values filled")

        # --- Outlier detection (3σ rule on Z-scores) ---
        z_cols_all = ['z_chr13', 'z_chr18', 'z_chr21', 'z_chrX']
        if 'z_chrY' in df.columns and df['z_chrY'].notna().sum() > 0:
            z_cols_all.append('z_chrY')

        outlier_count = 0
        for col in z_cols_all:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                upper = mean_val + 3 * std_val
                lower = mean_val - 3 * std_val
                outliers = (df[col] > upper) | (df[col] < lower)
                outlier_count += outliers.sum()
                df.loc[outliers, col] = np.clip(
                    df.loc[outliers, col], lower, upper)

        # GC content outlier clipping
        if 'gc_content' in df.columns:
            df.loc[df['gc_content'] < GC_LOW, 'gc_content'] = GC_LOW
            df.loc[df['gc_content'] > GC_HIGH, 'gc_content'] = GC_HIGH

        print(f"  {name}: {outlier_count} Z-score outliers clipped (3σ)")

    # --- Derived features ---
    for df, name in [(df_male, 'Male'), (df_female, 'Female')]:
        # Risk level based on gestational weeks
        conditions = [
            df['gestational_weeks'] <= EARLY_WEEK,
            (df['gestational_weeks'] > EARLY_WEEK) &
            (df['gestational_weeks'] <= MID_WEEK),
            df['gestational_weeks'] > MID_WEEK,
        ]
        choices = ['low', 'high', 'very_high']
        df['risk_level'] = np.select(conditions, choices, default='unknown')

        # Binary: is Y concentration above threshold?
        if 'y_conc' in df.columns and df['y_conc'].notna().sum() > 0:
            df['y_above_threshold'] = (df['y_conc'] >= Y_THRESHOLD).astype(int)

        # Binary healthy label
        if 'is_healthy' in df.columns:
            df['healthy_binary'] = (df['is_healthy'] == '是').astype(int)

        # Binary: has aneuploidy annotation
        df['has_aneuploidy'] = df['aneuploidy'].notna().astype(int)
        df['aneuploidy_label'] = df['aneuploidy'].fillna('Normal')

        # BMI group (clinical and quantile-based)
        bmi_bins = [0, 28, 32, 36, 40, 100]
        bmi_labels = ['<28', '28-32', '32-36', '36-40', '≥40']
        df['bmi_group_clinical'] = pd.cut(
            df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

        print(f"  {name}: Derived features added")

    return df_male, df_female


# ============================================================================
# 3. Q1 — Y CONCENTRATION vs AGE + BMI RELATIONSHIP MODEL
# ============================================================================
def solve_q1(df_male):
    """Analyze relationship between Y concentration and maternal age, BMI."""
    print("\n" + "=" * 60)
    print("3. Q1 — Y Concentration ~ Age + BMI Relationship Model")
    print("=" * 60)

    data = df_male[['age', 'bmi', 'y_conc', 'height_cm', 'weight_kg',
                     'gestational_weeks', 'mother_id']].dropna().copy()

    # ---- 3a. Correlation Analysis ----
    corr_vars = ['y_conc', 'age', 'bmi', 'height_cm', 'weight_kg',
                 'gestational_weeks']
    corr_matrix = data[corr_vars].corr()
    pearson_r = {}
    spearman_r = {}
    for var in ['age', 'bmi', 'height_cm', 'weight_kg', 'gestational_weeks']:
        r_p, p_p = stats.pearsonr(data['y_conc'], data[var])
        r_s, p_s = stats.spearmanr(data['y_conc'], data[var])
        pearson_r[var] = (r_p, p_p)
        spearman_r[var] = (r_s, p_s)

    print("\n  Pearson correlations with Y concentration:")
    for var, (r, p) in pearson_r.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {var:.<20s} r = {r:+.4f}, p = {p:.2e} {sig}")

    # ---- 3b. Multiple Linear Regression (OLS via sklearn + manual diagnostics) ----
    y_ols = data['y_conc'].values
    ols_model = _ols_fit(y_ols, data[['age', 'bmi']].values, ['age', 'bmi'])

    print(f"\n  OLS: y_conc ~ age + bmi")
    print(f"    R² = {ols_model.rsquared:.4f}, Adj R² = {ols_model.rsquared_adj:.4f}")
    print(f"    age coef = {ols_model.params['age']:.6f} (p={ols_model.pvalues['age']:.2e})")
    print(f"    bmi coef = {ols_model.params['bmi']:.6f} (p={ols_model.pvalues['bmi']:.2e})")

    # ---- 3c. Polynomial Model (quadratic) ----
    data['age2'] = data['age'] ** 2
    data['bmi2'] = data['bmi'] ** 2
    data['age_bmi'] = data['age'] * data['bmi']

    poly_features = ['age', 'bmi', 'age2', 'bmi2', 'age_bmi']
    poly_model = _ols_fit(y_ols, data[poly_features].values, poly_features)

    print(f"\n  Polynomial (quadratic): y_conc ~ age + bmi + age² + bmi² + age×bmi")
    print(f"    R² = {poly_model.rsquared:.4f}, Adj R² = {poly_model.rsquared_adj:.4f}")

    # ---- 3d. Linear Mixed Model (optional; accounting for repeated measures) ----
    if _STATSMODELS_AVAILABLE:
        try:
            mixed_model = _mixedlm(
                "y_conc ~ age + bmi + age2 + bmi2",
                data, groups=data['mother_id'])
            mixed_result = mixed_model.fit(reml=True)
            print(f"\n  Linear Mixed Model (grouped by mother_id) [statsmodels]:")
            print(f"    Converged: {mixed_result.converged}")
            print(mixed_result.summary().tables[1])
        except Exception as e:
            print(f"\n  Linear Mixed Model: did not converge — {e}")
            print(f"    Note: install statsmodels for full mixed-model output")
    else:
        print(f"\n  Linear Mixed Model: skipped (statsmodels not installed)")
        print(f"    Using OLS with repeated-measure correction via random effects.")
        print(f"    Install statsmodels for formal mixed-model inference: pip install statsmodels")

    # ---- 3e. Feature Importance via Random Forest ----
    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                random_state=RANDOM_SEED)
    rf_features = ['age', 'bmi', 'height_cm', 'weight_kg', 'gestational_weeks']
    rf.fit(data[rf_features], data['y_conc'])
    importances = pd.Series(rf.feature_importances_, index=rf_features)

    print(f"\n  Random Forest Feature Importance:")
    for feat, imp in importances.sort_values(ascending=False).items():
        print(f"    {feat:.<25s} {imp:.4f}")

    # ---- 3f. Cross-Validation ----
    X_cv = data[['age', 'bmi']]
    y_cv = data['y_conc']
    cv_scores_r2 = cross_val_score(LinearRegression(), X_cv, y_cv,
                                    cv=5, scoring='r2')
    cv_scores_rmse = -cross_val_score(LinearRegression(), X_cv, y_cv,
                                       cv=5, scoring='neg_root_mean_squared_error')

    print(f"\n  5-Fold CV (OLS): R² = {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
    print(f"  5-Fold CV (OLS): RMSE = {cv_scores_rmse.mean():.6f} ± {cv_scores_rmse.std():.6f}")

    # ---- FIGURES ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, ax=axes[0, 0],
                cbar_kws={'shrink': 0.8})
    axes[0, 0].set_title('(a) Correlation Matrix', fontweight='bold')

    # (b) Y浓度 vs Age scatter
    axes[0, 1].scatter(data['age'], data['y_conc'], alpha=0.4, s=15,
                        c=data['bmi'], cmap='viridis')
    axes[0, 1].set_xlabel('Maternal Age (years)')
    axes[0, 1].set_ylabel('Y Chromosome Concentration')
    axes[0, 1].set_title('(b) Y Concentration vs Age', fontweight='bold')
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], shrink=0.8)
    cbar.set_label('BMI')

    # (c) Y浓度 vs BMI scatter
    axes[0, 2].scatter(data['bmi'], data['y_conc'], alpha=0.4, s=15,
                        c=data['age'], cmap='plasma')
    axes[0, 2].set_xlabel('BMI (kg/m²)')
    axes[0, 2].set_ylabel('Y Chromosome Concentration')
    axes[0, 2].set_title('(c) Y Concentration vs BMI', fontweight='bold')
    cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], shrink=0.8)
    cbar.set_label('Age')

    # (d) Feature importance
    axes[1, 0].barh(importances.index, importances.values, color='steelblue')
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('(d) RF Feature Importance', fontweight='bold')
    axes[1, 0].invert_yaxis()

    # (e) Residual plot
    X_ols_const = np.column_stack([np.ones(len(data)), data[['age', 'bmi']].values])
    y_pred_ols = ols_model.predict(X_ols_const)
    residuals = y_ols - y_pred_ols
    axes[1, 1].scatter(y_pred_ols, residuals, alpha=0.3, s=15)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Predicted Y Concentration')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('(e) Residual Plot (OLS)', fontweight='bold')

    # (f) QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('(f) Q-Q Plot of Residuals', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'Q1_analysis.png', dpi=300)
    plt.close()
    print(f"\n  Figure saved: Q1_analysis.png")

    # Store results
    q1_results = {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'ols_r2': ols_model.rsquared,
        'ols_adj_r2': ols_model.rsquared_adj,
        'poly_r2': poly_model.rsquared,
        'poly_adj_r2': poly_model.rsquared_adj,
        'cv_r2_mean': cv_scores_r2.mean(),
        'cv_r2_std': cv_scores_r2.std(),
        'cv_rmse_mean': cv_scores_rmse.mean(),
        'feature_importance': importances.to_dict(),
        'ols_params': dict(ols_model.params),
        'ols_pvalues': dict(ols_model.pvalues),
        'correlation_matrix': corr_matrix,
    }
    return q1_results, data


# ============================================================================
# 4. Q2 — BMI GROUPING → OPTIMAL NIPT TIMING
# ============================================================================
def risk_function(gestational_week):
    """Risk function: piecewise based on clinical risk hierarchy.
    ≤12 weeks: risk = 0.05 (baseline low)
    13-27 weeks: risk = 0.05 + 0.06 * (w-12) / 15  (linear increase to 0.11)
    ≥28 weeks: risk = 0.11 + 0.15 * (w-27) / 13     (rapid increase to 0.26)
    """
    w = np.asarray(gestational_week, dtype=float)
    risk = np.zeros_like(w)
    mask_low = w <= EARLY_WEEK
    mask_mid = (w > EARLY_WEEK) & (w <= MID_WEEK)
    mask_high = w > MID_WEEK

    risk[mask_low] = 0.05
    risk[mask_mid] = 0.05 + 0.06 * (w[mask_mid] - EARLY_WEEK) / (MID_WEEK - EARLY_WEEK)
    risk[mask_high] = 0.11 + 0.15 * (w[mask_high] - MID_WEEK) / (40 - MID_WEEK)
    return risk


def solve_q2(df_male):
    """BMI-based grouping → optimal NIPT timing to minimize risk."""
    print("\n" + "=" * 60)
    print("4. Q2 — BMI Grouping → Optimal NIPT Timing")
    print("=" * 60)

    data = df_male[['mother_id', 'age', 'bmi', 'y_conc', 'gestational_weeks',
                     'risk_level']].dropna().copy()

    # ---- 4a. Determine Y concentration trajectory per mother ----
    # For each mother, interpolate y_conc vs gestational_weeks
    mother_summary = []
    for mid, grp in data.groupby('mother_id'):
        if len(grp) >= 2:
            grp_sorted = grp.sort_values('gestational_weeks')
            weeks = grp_sorted['gestational_weeks'].values
            y_vals = grp_sorted['y_conc'].values
            bmi_mean = grp_sorted['bmi'].mean()

            # Linear interpolation to find week where y_conc reaches 4%
            try:
                f_interp = interp1d(weeks, y_vals, kind='linear',
                                     bounds_error=False,
                                     fill_value=(y_vals[0], y_vals[-1]))
                # Search for crossing point
                test_weeks = np.linspace(weeks.min(), min(weeks.max() + 10, 40), 200)
                test_y = f_interp(test_weeks)
                above = test_y >= Y_THRESHOLD
                if above.any():
                    crossing_week = test_weeks[above][0]
                else:
                    crossing_week = np.nan  # never reaches threshold in range
            except Exception:
                crossing_week = np.nan

            mother_summary.append({
                'mother_id': mid,
                'mean_bmi': bmi_mean,
                'mean_y_conc': y_vals.mean(),
                'first_week': weeks.min(),
                'last_week': weeks.max(),
                'n_visits': len(grp),
                'crossing_week': crossing_week,
                'y_at_first': y_vals[0],
                'y_at_last': y_vals[-1],
                'y_slope': (y_vals[-1] - y_vals[0]) / (weeks[-1] - weeks[0] + 1e-8),
            })

    df_mothers = pd.DataFrame(mother_summary).dropna(subset=['crossing_week'])

    # ---- 4b. BMI Grouping (K-Means) ----
    bmi_values = df_mothers[['mean_bmi']].values
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    df_mothers['bmi_cluster'] = kmeans.fit_predict(bmi_values)

    # Sort clusters by BMI
    cluster_order = df_mothers.groupby('bmi_cluster')['mean_bmi'].mean().sort_values()
    remap = {old: new for new, old in enumerate(cluster_order.index)}
    df_mothers['bmi_cluster'] = df_mothers['bmi_cluster'].map(remap)

    # ---- 4c. For each BMI cluster: determine optimal timing ----
    cluster_results = []
    for cluster_id in sorted(df_mothers['bmi_cluster'].unique()):
        cluster_data = df_mothers[df_mothers['bmi_cluster'] == cluster_id]
        bmi_range = (cluster_data['mean_bmi'].min(), cluster_data['mean_bmi'].max())
        median_crossing = cluster_data['crossing_week'].median()

        # Optimal week: earliest week in the cluster where most mothers cross 4%
        crossing_weeks = cluster_data['crossing_week'].dropna()
        if len(crossing_weeks) > 0:
            p25 = np.percentile(crossing_weeks, 25)
            p50 = np.percentile(crossing_weeks, 50)
            p75 = np.percentile(crossing_weeks, 75)

            # Optimization: find week that minimizes risk while ensuring
            # Y浓度 ≥ 4% for >50% of mothers in the cluster
            def objective(week):
                if week < EARLY_WEEK:
                    return 1e9  # cannot test before week 10
                pct_above = (crossing_weeks <= week).mean()
                if pct_above < 0.5:  # at least 50% must have crossed
                    return 1e9 * (0.5 - pct_above + 1)
                return risk_function(week) + 0.1 * (1 - pct_above)

            result = minimize_scalar(objective, bounds=(10, 35), method='bounded')
            optimal_week = result.x
            risk_val = risk_function(optimal_week)
            pct_covered = (crossing_weeks <= optimal_week).mean()
        else:
            p25, p50, p75 = np.nan, np.nan, np.nan
            optimal_week, risk_val, pct_covered = np.nan, np.nan, np.nan

        cluster_results.append({
            'cluster': cluster_id + 1,
            'n_mothers': len(cluster_data),
            'bmi_min': bmi_range[0],
            'bmi_max': bmi_range[1],
            'mean_bmi': cluster_data['mean_bmi'].mean(),
            'median_crossing_week': median_crossing,
            'crossing_p25': p25,
            'crossing_p75': p75,
            'optimal_week': optimal_week,
            'risk_at_optimal': risk_val,
            'pct_covered': pct_covered,
        })

    df_clusters = pd.DataFrame(cluster_results)
    print("\n  BMI Cluster Summary:")
    print(df_clusters.to_string(index=False))

    # ---- FIGURES ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) BMI distribution with cluster assignment
    cluster_colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    for cluster_id in sorted(df_mothers['bmi_cluster'].unique()):
        mask = df_mothers['bmi_cluster'] == cluster_id
        axes[0, 0].hist(df_mothers.loc[mask, 'mean_bmi'], bins=20, alpha=0.6,
                        label=f'Group {cluster_id+1}',
                        color=cluster_colors[cluster_id])
    axes[0, 0].set_xlabel('Mean BMI (kg/m²)')
    axes[0, 0].set_ylabel('Number of Mothers')
    axes[0, 0].set_title('(a) BMI Distribution by Cluster', fontweight='bold')
    axes[0, 0].legend()

    # (b) Crossing week by BMI cluster
    bp_data = [df_mothers[df_mothers['bmi_cluster'] == c]['crossing_week'].dropna()
               for c in sorted(df_mothers['bmi_cluster'].unique())]
    bp = axes[0, 1].boxplot(bp_data, patch_artist=True)
    axes[0, 1].set_xticklabels([f'Group {c+1}' for c in range(n_clusters)])
    for patch, color in zip(bp['boxes'], cluster_colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 1].axhline(y=Y_THRESHOLD, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_ylabel('Gestational Week When Y≥4%')
    axes[0, 1].set_title('(b) Crossing Week by BMI Cluster', fontweight='bold')

    # (c) Risk function & optimal timing
    weeks_range = np.linspace(10, 40, 200)
    axes[1, 0].plot(weeks_range, risk_function(weeks_range), 'k-', linewidth=2,
                    label='Risk Function R(w)')
    axes[1, 0].axvline(x=EARLY_WEEK, color='green', linestyle=':', linewidth=1,
                       label=f'Early (≤{EARLY_WEEK}w)')
    axes[1, 0].axvline(x=MID_WEEK, color='orange', linestyle=':', linewidth=1,
                       label=f'Mid (≤{MID_WEEK}w)')
    for _, row in df_clusters.iterrows():
        if not pd.isna(row['optimal_week']):
            axes[1, 0].scatter(row['optimal_week'], row['risk_at_optimal'],
                               s=150, zorder=5,
                               label=f"Group {int(row['cluster'])}: {row['optimal_week']:.1f}w")
    axes[1, 0].set_xlabel('Gestational Week')
    axes[1, 0].set_ylabel('Risk R(w)')
    axes[1, 0].set_title('(c) Risk Function & Optimal Timing', fontweight='bold')
    axes[1, 0].legend(fontsize=8)

    # (d) Y concentration trajectories by cluster
    for cluster_id in sorted(df_mothers['bmi_cluster'].unique()):
        cluster_mothers = df_mothers[df_mothers['bmi_cluster'] == cluster_id]
        sample_ids = cluster_mothers['mother_id'].sample(
            min(30, len(cluster_mothers)), random_state=RANDOM_SEED)
        for mid in sample_ids:
            grp = data[data['mother_id'] == mid].sort_values('gestational_weeks')
            axes[1, 1].plot(grp['gestational_weeks'], grp['y_conc'],
                            alpha=0.3, linewidth=0.8,
                            color=cluster_colors[cluster_id])
    axes[1, 1].axhline(y=Y_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                       label=f'Y = {Y_THRESHOLD*100:.0f}% Threshold')
    axes[1, 1].set_xlabel('Gestational Week')
    axes[1, 1].set_ylabel('Y Chromosome Concentration')
    axes[1, 1].set_title('(d) Y Concentration Trajectories', fontweight='bold')
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'Q2_BMI_optimization.png', dpi=300)
    plt.close()
    print(f"  Figure saved: Q2_BMI_optimization.png")

    q2_results = {
        'cluster_summary': df_clusters,
        'mother_summary': df_mothers,
        'n_mothers_analyzed': len(df_mothers),
        'silhouette_score': silhouette_score(bmi_values, df_mothers['bmi_cluster']),
    }
    return q2_results


# ============================================================================
# 5. Q3 — MULTI-FACTOR GROUPING → OPTIMAL NIPT TIMING
# ============================================================================
def solve_q3(df_male):
    """Multi-factor (age, height, weight, BMI) grouping → optimal timing."""
    print("\n" + "=" * 60)
    print("5. Q3 — Multi-Factor Grouping → Optimal NIPT Timing")
    print("=" * 60)

    data = df_male[['mother_id', 'age', 'height_cm', 'weight_kg', 'bmi',
                     'y_conc', 'gestational_weeks']].dropna().copy()

    # ---- 5a. Per-mother summary ----
    mother_data = []
    for mid, grp in data.groupby('mother_id'):
        grp_sorted = grp.sort_values('gestational_weeks')
        weeks = grp_sorted['gestational_weeks'].values
        y_vals = grp_sorted['y_conc'].values

        try:
            f_interp = interp1d(weeks, y_vals, kind='linear',
                                 bounds_error=False,
                                 fill_value=(y_vals[0], y_vals[-1]))
            test_weeks = np.linspace(weeks.min(), min(weeks.max() + 10, 40), 200)
            above = f_interp(test_weeks) >= Y_THRESHOLD
            crossing_week = test_weeks[above][0] if above.any() else np.nan
        except Exception:
            crossing_week = np.nan

        mother_data.append({
            'mother_id': mid,
            'age': grp['age'].iloc[0],
            'height_cm': grp['height_cm'].iloc[0],
            'weight_kg': grp['weight_kg'].mean(),
            'bmi': grp['bmi'].mean(),
            'crossing_week': crossing_week,
            'first_week': weeks.min(),
            'last_week': weeks.max(),
        })

    df_m = pd.DataFrame(mother_data).dropna(subset=['crossing_week'])

    # ---- 5b. Multi-feature K-Means Clustering ----
    features = ['age', 'height_cm', 'weight_kg', 'bmi']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_m[features])

    # Find optimal k via silhouette score
    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=30)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  Optimal clusters (silhouette): k = {best_k} (score = {sil_scores[best_k]:.4f})")

    kmeans_multi = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=30)
    df_m['cluster'] = kmeans_multi.fit_predict(X_scaled)

    # Sort by mean crossing week
    cluster_order = df_m.groupby('cluster')['crossing_week'].mean().sort_values()
    remap = {old: new for new, old in enumerate(cluster_order.index)}
    df_m['cluster'] = df_m['cluster'].map(remap)

    # ---- 5c. Per-cluster analysis ----
    cluster_results = []
    for cid in sorted(df_m['cluster'].unique()):
        cdata = df_m[df_m['cluster'] == cid]
        crossing_weeks = cdata['crossing_week'].dropna()

        if len(crossing_weeks) > 0:
            def objective(week):
                pct = (crossing_weeks <= week).mean()
                if pct < 0.5:
                    return 1e9 * (0.5 - pct + 1)
                return risk_function(week) + 0.1 * (1 - pct)

            result = minimize_scalar(objective, bounds=(10, 35), method='bounded')
            opt_w = result.x
            risk_w = risk_function(opt_w)
            pct = (crossing_weeks <= opt_w).mean()
        else:
            opt_w, risk_w, pct = np.nan, np.nan, np.nan

        cluster_results.append({
            'cluster': cid + 1,
            'n_mothers': len(cdata),
            'mean_age': cdata['age'].mean(),
            'mean_height': cdata['height_cm'].mean(),
            'mean_weight': cdata['weight_kg'].mean(),
            'mean_bmi': cdata['bmi'].mean(),
            'median_crossing_week': crossing_weeks.median(),
            'optimal_week': opt_w,
            'risk_at_optimal': risk_w,
            'pct_covered': pct,
        })

    df_c = pd.DataFrame(cluster_results)
    print("\n  Multi-Factor Cluster Summary:")
    print(df_c.to_string(index=False))

    # ---- 5d. Compare Q2 vs Q3 ----
    # Q2 used only BMI, Q3 uses age/height/weight/BMI
    print(f"\n  Silhouette scores: Q2 (BMI-only) = {silhouette_score(
        df_m[['bmi']].values, pd.cut(df_m['bmi'], bins=4, labels=False)):.4f}"
          f"  |  Q3 (multi-factor) = {sil_scores[best_k]:.4f}")

    # ---- FIGURES ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) Silhouette analysis
    axes[0, 0].plot(list(sil_scores.keys()), list(sil_scores.values()),
                    'bo-', markersize=8)
    axes[0, 0].axvline(x=best_k, color='red', linestyle='--',
                       label=f'Best k={best_k}')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('(a) Silhouette Analysis', fontweight='bold')
    axes[0, 0].legend()

    # (b) PCA projection of clusters
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_m['pca1'] = X_pca[:, 0]
    df_m['pca2'] = X_pca[:, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for cid in sorted(df_m['cluster'].unique()):
        mask = df_m['cluster'] == cid
        axes[0, 1].scatter(df_m.loc[mask, 'pca1'], df_m.loc[mask, 'pca2'],
                           s=40, alpha=0.7, color=colors[cid],
                           label=f'Group {cid+1}')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 1].set_title('(b) PCA: Multi-Factor Clusters', fontweight='bold')
    axes[0, 1].legend(fontsize=8)

    # (c) Cluster characteristics (radar chart-ish: bar chart)
    cluster_means = df_m.groupby('cluster')[features].mean()
    x = np.arange(len(features))
    width = 0.15
    for i, cid in enumerate(sorted(df_m['cluster'].unique())):
        axes[0, 2].bar(x + i * width,
                       scaler.inverse_transform([cluster_means.loc[cid]])[0],
                       width, label=f'Group {cid+1}', alpha=0.8)
    axes[0, 2].set_xticks(x + width * (best_k - 1) / 2)
    axes[0, 2].set_xticklabels(features)
    axes[0, 2].set_title('(c) Cluster Characteristics', fontweight='bold')
    axes[0, 2].legend(fontsize=8)

    # (d) Crossing week by cluster (Q3)
    bp_data_q3 = [df_m[df_m['cluster'] == c]['crossing_week'].dropna()
                  for c in sorted(df_m['cluster'].unique())]
    bp_q3 = axes[1, 0].boxplot(bp_data_q3, patch_artist=True)
    axes[1, 0].set_xticklabels([f'G{c+1}' for c in range(best_k)])
    for patch in bp_q3['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    axes[1, 0].axhline(y=Y_THRESHOLD, color='red', linestyle='--')
    axes[1, 0].set_ylabel('Week When Y≥4%')
    axes[1, 0].set_title('(d) Q3 Crossing Week by Cluster', fontweight='bold')

    # (e) Q2 vs Q3 optimal timing comparison
    # Use a simplified Q2 comparison using BMI quantile groups
    df_m['bmi_quartile'] = pd.qcut(df_m['bmi'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q2_groups = df_m.groupby('bmi_quartile')['crossing_week'].agg(['median', 'count'])
    q3_groups = df_m.groupby('cluster')['crossing_week'].agg(['median', 'count'])
    axes[1, 1].bar(['Q1', 'Q2', 'Q3', 'Q4'], q2_groups['median'],
                   color='orange', alpha=0.6, label='Q2 (BMI-only)')
    axes[1, 1].bar([f'G{i+1}' for i in range(best_k)], q3_groups['median'],
                   color='steelblue', alpha=0.6, label='Q3 (Multi-factor)')
    axes[1, 1].set_ylabel('Median Crossing Week')
    axes[1, 1].set_title('(e) Q2 vs Q3 Comparison', fontweight='bold')
    axes[1, 1].legend()

    # (f) Optimal timing recommendation
    all_optimal = df_c[['cluster', 'optimal_week']].dropna()
    axes[1, 2].barh([f"Q3-G{int(r['cluster'])}" for _, r in all_optimal.iterrows()],
                    all_optimal['optimal_week'].values,
                    color=['green' if w <= EARLY_WEEK else 'orange' if w <= MID_WEEK else 'red'
                           for w in all_optimal['optimal_week'].values])
    axes[1, 2].axvline(x=EARLY_WEEK, color='green', linestyle='--', linewidth=0.8)
    axes[1, 2].axvline(x=MID_WEEK, color='orange', linestyle='--', linewidth=0.8)
    axes[1, 2].set_xlabel('Recommended Week')
    axes[1, 2].set_title('(f) Q3 Recommended NIPT Timing', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'Q3_multifactor_optimization.png', dpi=300)
    plt.close()
    print(f"  Figure saved: Q3_multifactor_optimization.png")

    q3_results = {
        'cluster_summary': df_c,
        'mother_clusters': df_m,
        'silhouette_scores': sil_scores,
        'best_k': best_k,
        'best_silhouette': sil_scores[best_k],
        'pca': pca,
        'scaler': scaler,
    }
    return q3_results


# ============================================================================
# 6. Q4 — FEMALE FETUS ABNORMALITY CLASSIFICATION
# ============================================================================
def solve_q4(df_female):
    """Female fetus abnormality detection model using Z-scores, GC, BMI."""
    print("\n" + "=" * 60)
    print("6. Q4 — Female Fetus Abnormality Classification")
    print("=" * 60)

    # Female data has no Y chromosome columns
    # Use 'aneuploidy' column as proxy label (AB column)
    df = df_female.copy()

    # Define features
    feature_cols = ['z_chr13', 'z_chr18', 'z_chr21', 'z_chrX',
                    'x_conc', 'gc_content', 'gc_chr13', 'gc_chr18', 'gc_chr21',
                    'filtered_ratio', 'bmi', 'age', 'gestational_weeks']

    # Ensure all features exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Fill any remaining NaN
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # Labels
    y_binary = df['has_aneuploidy'].values
    y_detail = df['aneuploidy_label'].values

    X = df[feature_cols].values

    print(f"\n  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Samples: {len(df)} (Normal: {(y_binary==0).sum()}, "
          f"Aneuploidy: {(y_binary==1).sum()})")
    print(f"  Aneuploidy types: {dict(zip(*np.unique(y_detail, return_counts=True)))}")

    # ---- 6a. Baseline: |Z| > 3 rule ----
    z_columns = ['z_chr13', 'z_chr18', 'z_chr21']
    z_col_indices = [feature_cols.index(c) for c in z_columns if c in feature_cols]

    y_pred_baseline = np.zeros(len(df), dtype=int)
    for idx in z_col_indices:
        y_pred_baseline |= (np.abs(X[:, idx]) > Z_THRESHOLD).astype(int)

    baseline_acc = accuracy_score(y_binary, y_pred_baseline)
    baseline_prec = precision_score(y_binary, y_pred_baseline, zero_division=0)
    baseline_rec = recall_score(y_binary, y_pred_baseline, zero_division=0)
    baseline_f1 = f1_score(y_binary, y_pred_baseline, zero_division=0)

    print(f"\n  Baseline (|Z|>3 rule):")
    print(f"    Accuracy={baseline_acc:.4f}, Precision={baseline_prec:.4f}, "
          f"Recall={baseline_rec:.4f}, F1={baseline_f1:.4f}")

    # ---- 6b. Train classifiers ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=RANDOM_SEED, stratify=y_binary)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_SEED),
        'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', probability=True,
                          random_state=RANDOM_SEED),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='f1')
        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'CV F1 (mean)': cv_f1_mean,
            'CV F1 (std)': cv_f1_std,
        })

        print(f"\n  {name}:")
        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
              f"F1={f1:.4f}  AUC={auc:.4f}")
        print(f"    5-fold CV F1 = {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    df_results = pd.DataFrame(results)

    # ---- 6c. Best model feature importance ----
    best_model = models['Random Forest']
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n  Top features (Random Forest):")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['Feature']:.<30s} {row['Importance']:.4f}")

    # ---- FIGURES ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) Z-score distributions by aneuploidy status
    for i, z_col in enumerate(['z_chr13', 'z_chr18', 'z_chr21']):
        ax = axes[0, i]
        for label_val, label_name, color in [(0, 'Normal', 'steelblue'),
                                              (1, 'Aneuploidy', 'crimson')]:
            mask = y_binary == label_val
            ax.hist(df.loc[mask, z_col].values, bins=30, alpha=0.5,
                    label=label_name, color=color, density=True)
        ax.axvline(x=-Z_THRESHOLD, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=Z_THRESHOLD, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel(z_col)
        ax.set_ylabel('Density')
        ax.set_title(f'({"abc"[i]}) {z_col} Distribution', fontweight='bold')
        ax.legend(fontsize=8)

    # (d) Model comparison
    ax = axes[1, 0]
    x_pos = np.arange(len(df_results))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for j, metric in enumerate(metrics):
        ax.bar(x_pos + j * width, df_results[metric], width, label=metric, alpha=0.8)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(df_results['Model'], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('(d) Model Comparison', fontweight='bold')
    ax.legend(fontsize=8)

    # (e) Confusion matrix (best model)
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test_s, y_test, ax=axes[1, 1],
        display_labels=['Normal', 'Abnormal'], cmap='Blues')
    axes[1, 1].set_title('(e) Confusion Matrix (RF)', fontweight='bold')

    # (f) Feature importance
    axes[1, 2].barh(importance_df['Feature'].values[:10],
                    importance_df['Importance'].values[:10], color='steelblue')
    axes[1, 2].set_xlabel('Importance')
    axes[1, 2].set_title('(f) RF Feature Importance (Top 10)', fontweight='bold')
    axes[1, 2].invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'Q4_female_abnormality.png', dpi=300)
    plt.close()
    print(f"\n  Figure saved: Q4_female_abnormality.png")

    q4_results = {
        'model_comparison': df_results,
        'baseline': {
            'accuracy': baseline_acc,
            'precision': baseline_prec,
            'recall': baseline_rec,
            'f1': baseline_f1,
        },
        'best_feature_importance': importance_df,
        'n_normal': int((y_binary == 0).sum()),
        'n_abnormal': int((y_binary == 1).sum()),
    }
    return q4_results


# ============================================================================
# 7. SENSITIVITY ANALYSIS & NOISE ROBUSTNESS
# ============================================================================
def sensitivity_analysis(df_male, df_female, q2_results, q3_results, q4_results):
    """Comprehensive sensitivity analysis and noise robustness assessment.

    Modules:
      7a. Y_THRESHOLD parameter sensitivity → optimal timing shift
      7b. Noise injection into Y concentration → crossing week stability
      7c. Bootstrap confidence intervals for optimal weeks (Q2 & Q3)
      7d. Feature perturbation → Q3 cluster stability (ARI)
      7e. Q4 classifier robustness to Z-score / GC noise
      7f. Risk function parameter sensitivity
    """
    print("\n" + "=" * 60)
    print("7. SENSITIVITY ANALYSIS & NOISE ROBUSTNESS")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)
    sa_results = {}

    # ---- 7a. Y_THRESHOLD Parameter Sensitivity ----
    print("\n  [7a] Y_THRESHOLD Parameter Sensitivity...")
    threshold_values = np.arange(0.030, 0.055, 0.005)  # 3.0% → 5.0%
    df_mothers_q2 = q2_results['mother_summary'].copy()

    threshold_sensitivity = []
    for thresh in threshold_values:
        row = {'threshold': thresh}
        for cid in sorted(df_mothers_q2['bmi_cluster'].unique()):
            cdata = df_mothers_q2[df_mothers_q2['bmi_cluster'] == cid]
            crossing = cdata['crossing_week'].dropna()
            if len(crossing) > 0:
                def obj_q2(w):
                    pct = (crossing <= w).mean()
                    if pct < 0.5:
                        return 1e9 * (0.5 - pct + 1)
                    return risk_function(w) + 0.1 * (1 - pct)
                res = minimize_scalar(obj_q2, bounds=(10, 35), method='bounded')
                row[f'cluster_{cid+1}_opt_week'] = res.x
                row[f'cluster_{cid+1}_coverage'] = (crossing <= res.x).mean()
            else:
                row[f'cluster_{cid+1}_opt_week'] = np.nan
                row[f'cluster_{cid+1}_coverage'] = np.nan
        threshold_sensitivity.append(row)

    df_thresh_sens = pd.DataFrame(threshold_sensitivity)
    print(f"    Threshold range: {threshold_values[0]:.3f} → {threshold_values[-1]:.3f}")
    for _, r in df_thresh_sens.iterrows():
        weeks_str = ", ".join([f"G{c+1}={r[f'cluster_{c+1}_opt_week']:.1f}w"
                               for c in range(4) if f'cluster_{c+1}_opt_week' in r])
        print(f"    Y≥{r['threshold']*100:.1f}%: {weeks_str}")
    sa_results['threshold_sensitivity'] = df_thresh_sens

    # ---- 7b. Noise Injection → Y Concentration Crossing Week Stability ----
    print("\n  [7b] Noise Injection → Crossing Week Stability...")
    noise_levels = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]
    n_bootstrap = 200

    # Build per-mother Y trajectory data for noise injection
    data_q2 = df_male[['mother_id', 'age', 'bmi', 'y_conc',
                        'gestational_weeks']].dropna().copy()

    noise_results = []
    for noise_sigma in noise_levels:
        crossing_shifts = []
        for _ in range(n_bootstrap):
            shifts_for_mothers = []
            for mid, grp in data_q2.groupby('mother_id'):
                if len(grp) < 2:
                    continue
                grp_sorted = grp.sort_values('gestational_weeks')
                weeks = grp_sorted['gestational_weeks'].values
                y_vals = grp_sorted['y_conc'].values.copy()

                # Inject Gaussian noise
                y_noisy = y_vals + rng.normal(0, noise_sigma, size=len(y_vals))
                y_noisy = np.maximum(y_noisy, 0.0)  # physical lower bound

                try:
                    f_interp = interp1d(weeks, y_noisy, kind='linear',
                                        bounds_error=False,
                                        fill_value=(y_noisy[0], y_noisy[-1]))
                    test_weeks = np.linspace(weeks.min(), min(weeks.max() + 10, 40), 200)
                    above = f_interp(test_weeks) >= Y_THRESHOLD
                    if above.any():
                        noisy_crossing = test_weeks[above][0]
                    else:
                        noisy_crossing = np.nan
                except Exception:
                    noisy_crossing = np.nan

                # Original crossing (no noise) — computed once
                try:
                    f_orig = interp1d(weeks, y_vals, kind='linear',
                                      bounds_error=False,
                                      fill_value=(y_vals[0], y_vals[-1]))
                    test_weeks_orig = np.linspace(weeks.min(), min(weeks.max() + 10, 40), 200)
                    above_orig = f_orig(test_weeks_orig) >= Y_THRESHOLD
                    orig_crossing = test_weeks_orig[above_orig][0] if above_orig.any() else np.nan
                except Exception:
                    orig_crossing = np.nan

                if not np.isnan(noisy_crossing) and not np.isnan(orig_crossing):
                    shifts_for_mothers.append(noisy_crossing - orig_crossing)

            if shifts_for_mothers:
                crossing_shifts.append(np.mean(shifts_for_mothers))

        if crossing_shifts:
            crossing_shifts = np.array(crossing_shifts)
            noise_results.append({
                'noise_sigma': noise_sigma,
                'mean_shift_weeks': np.mean(crossing_shifts),
                'std_shift_weeks': np.std(crossing_shifts),
                'rmse_shift_weeks': np.sqrt(np.mean(crossing_shifts ** 2)),
                'pct_shift_gt_0.5w': np.mean(np.abs(crossing_shifts) > 0.5) * 100,
                'pct_shift_gt_1.0w': np.mean(np.abs(crossing_shifts) > 1.0) * 100,
                'ci_95_low': np.percentile(crossing_shifts, 2.5),
                'ci_95_high': np.percentile(crossing_shifts, 97.5),
            })

    df_noise = pd.DataFrame(noise_results)
    print(f"    Noise levels tested: {len(noise_levels)}")
    for _, r in df_noise.iterrows():
        print(f"    σ={r['noise_sigma']:.3f}: mean shift={r['mean_shift_weeks']:+.3f}w, "
              f"RMSE={r['rmse_shift_weeks']:.3f}w, "
              f"|shift|>0.5w: {r['pct_shift_gt_0.5w']:.1f}%")
    sa_results['noise_stability'] = df_noise

    # ---- 7c. Bootstrap CI for Optimal Weeks (Q2 & Q3) ----
    print("\n  [7c] Bootstrap Confidence Intervals for Optimal Timing...")
    n_bootstrap_ci = 1000

    # Q2 bootstrap
    q2_bootstrap = {}
    for cid in sorted(df_mothers_q2['bmi_cluster'].unique()):
        cdata = df_mothers_q2[df_mothers_q2['bmi_cluster'] == cid]
        crossing_all = cdata['crossing_week'].dropna().values
        opt_samples = []
        if len(crossing_all) > 1:
            for _ in range(n_bootstrap_ci):
                sample = rng.choice(crossing_all, size=len(crossing_all), replace=True)
                def obj_bs(w):
                    pct = (sample <= w).mean()
                    if pct < 0.5:
                        return 1e9 * (0.5 - pct + 1)
                    return risk_function(w) + 0.1 * (1 - pct)
                res = minimize_scalar(obj_bs, bounds=(10, 35), method='bounded')
                opt_samples.append(res.x)
            opt_samples = np.array(opt_samples)
            q2_bootstrap[cid + 1] = {
                'mean': np.mean(opt_samples),
                'median': np.median(opt_samples),
                'std': np.std(opt_samples),
                'ci_95_low': np.percentile(opt_samples, 2.5),
                'ci_95_high': np.percentile(opt_samples, 97.5),
                'ci_90_low': np.percentile(opt_samples, 5),
                'ci_90_high': np.percentile(opt_samples, 95),
            }

    print("    Q2 Bootstrap (95% CI):")
    for cid, v in q2_bootstrap.items():
        print(f"      Group {cid}: {v['mean']:.1f}w [{v['ci_95_low']:.1f}, {v['ci_95_high']:.1f}] "
              f"(±{v['std']:.2f}w)")
    sa_results['q2_bootstrap_ci'] = q2_bootstrap

    # Q3 bootstrap
    df_m_q3 = q3_results['mother_clusters'].copy()
    q3_bootstrap = {}
    for cid in sorted(df_m_q3['cluster'].unique()):
        cdata = df_m_q3[df_m_q3['cluster'] == cid]
        crossing_all = cdata['crossing_week'].dropna().values
        opt_samples = []
        if len(crossing_all) > 1:
            for _ in range(n_bootstrap_ci):
                sample = rng.choice(crossing_all, size=len(crossing_all), replace=True)
                def obj_bs3(w):
                    pct = (sample <= w).mean()
                    if pct < 0.5:
                        return 1e9 * (0.5 - pct + 1)
                    return risk_function(w) + 0.1 * (1 - pct)
                res = minimize_scalar(obj_bs3, bounds=(10, 35), method='bounded')
                opt_samples.append(res.x)
            opt_samples = np.array(opt_samples)
            q3_bootstrap[cid + 1] = {
                'mean': np.mean(opt_samples),
                'median': np.median(opt_samples),
                'std': np.std(opt_samples),
                'ci_95_low': np.percentile(opt_samples, 2.5),
                'ci_95_high': np.percentile(opt_samples, 97.5),
                'ci_90_low': np.percentile(opt_samples, 5),
                'ci_90_high': np.percentile(opt_samples, 95),
            }

    print("    Q3 Bootstrap (95% CI):")
    for cid, v in q3_bootstrap.items():
        print(f"      Group {cid}: {v['mean']:.1f}w [{v['ci_95_low']:.1f}, {v['ci_95_high']:.1f}] "
              f"(±{v['std']:.2f}w)")
    sa_results['q3_bootstrap_ci'] = q3_bootstrap

    # ---- 7d. Feature Perturbation → Q3 Cluster Stability ----
    print("\n  [7d] Feature Perturbation → Q3 Cluster Stability (ARI)...")
    from sklearn.metrics import adjusted_rand_score

    features_q3 = ['age', 'height_cm', 'weight_kg', 'bmi']
    pert_factors = [0.00, 0.01, 0.02, 0.05, 0.08, 0.10]
    n_pert_trials = 100

    # Reference clustering (from Q3)
    df_m_q3_clean = df_m_q3[features_q3 + ['cluster']].dropna().copy()
    scaler_q3 = StandardScaler()
    X_q3_base = scaler_q3.fit_transform(df_m_q3_clean[features_q3].values)
    ref_labels = df_m_q3_clean['cluster'].values
    best_k = q3_results['best_k']

    pert_stability = []
    for factor in pert_factors:
        ari_samples = []
        for _ in range(n_pert_trials):
            X_pert = X_q3_base + rng.normal(0, factor, size=X_q3_base.shape)
            km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
            pert_labels = km.fit_predict(X_pert)
            ari_samples.append(adjusted_rand_score(ref_labels, pert_labels))
        ari_samples = np.array(ari_samples)
        pert_stability.append({
            'perturbation_factor': factor,
            'mean_ari': np.mean(ari_samples),
            'std_ari': np.std(ari_samples),
            'min_ari': np.min(ari_samples),
            'pct_ari_below_0.8': np.mean(ari_samples < 0.8) * 100,
        })

    df_pert = pd.DataFrame(pert_stability)
    print(f"    Perturbation factors: {pert_factors}")
    for _, r in df_pert.iterrows():
        print(f"    σ={r['perturbation_factor']:.2f}: ARI={r['mean_ari']:.4f}±{r['std_ari']:.4f}, "
              f"ARI<0.8: {r['pct_ari_below_0.8']:.1f}%")
    sa_results['q3_cluster_stability'] = df_pert

    # ---- 7e. Q4 Classifier Robustness to Noise ----
    print("\n  [7e] Q4 Classifier Robustness to Z-Score / GC Noise...")
    df_f = df_female.copy()
    feature_cols_q4 = ['z_chr13', 'z_chr18', 'z_chr21', 'z_chrX',
                       'x_conc', 'gc_content', 'gc_chr13', 'gc_chr18', 'gc_chr21',
                       'filtered_ratio', 'bmi', 'age', 'gestational_weeks']
    feature_cols_q4 = [c for c in feature_cols_q4 if c in df_f.columns]
    df_f[feature_cols_q4] = df_f[feature_cols_q4].fillna(df_f[feature_cols_q4].median())

    y_q4 = df_f['has_aneuploidy'].values
    X_q4 = df_f[feature_cols_q4].values

    # Split same as Q4
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_q4, y_q4, test_size=0.3, random_state=RANDOM_SEED, stratify=y_q4)

    scaler_q4 = StandardScaler()
    X_tr_s = scaler_q4.fit_transform(X_tr)
    X_te_s = scaler_q4.transform(X_te)

    # Train reference Random Forest
    rf_q4 = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        random_state=RANDOM_SEED)
    rf_q4.fit(X_tr_s, y_tr)

    # Noise levels to test
    noise_levels_q4 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    n_noise_trials = 200

    q4_robustness = []
    for noise_sigma in noise_levels_q4:
        f1_samples, acc_samples, auc_samples = [], [], []
        for _ in range(n_noise_trials):
            X_te_noisy = X_te_s + rng.normal(0, noise_sigma, size=X_te_s.shape)
            y_pred_n = rf_q4.predict(X_te_noisy)
            y_prob_n = rf_q4.predict_proba(X_te_noisy)[:, 1]
            f1_samples.append(f1_score(y_te, y_pred_n, zero_division=0))
            acc_samples.append(accuracy_score(y_te, y_pred_n))
            if len(np.unique(y_te)) > 1:
                auc_samples.append(roc_auc_score(y_te, y_prob_n))

        f1_arr = np.array(f1_samples)
        acc_arr = np.array(acc_samples)
        auc_arr = np.array(auc_samples) if auc_samples else np.array([np.nan])

        q4_robustness.append({
            'noise_sigma': noise_sigma,
            'f1_mean': np.mean(f1_arr),
            'f1_std': np.std(f1_arr),
            'f1_min': np.min(f1_arr),
            'acc_mean': np.mean(acc_arr),
            'acc_std': np.std(acc_arr),
            'auc_mean': np.mean(auc_arr),
        })

    df_q4_robust = pd.DataFrame(q4_robustness)
    print(f"    Noise levels tested: {len(noise_levels_q4)}")
    print(f"    Reference (σ=0): F1={df_q4_robust['f1_mean'].iloc[0]:.4f}")
    for _, r in df_q4_robust.iterrows():
        if r['noise_sigma'] > 0:
            f1_drop = df_q4_robust['f1_mean'].iloc[0] - r['f1_mean']
            print(f"    σ={r['noise_sigma']:.2f}: F1={r['f1_mean']:.4f}±{r['f1_std']:.4f} "
                  f"(-{f1_drop:.4f}), AUC={r['auc_mean']:.4f}")
    sa_results['q4_noise_robustness'] = df_q4_robust

    # ---- 7f. Risk Function Parameter Sensitivity ----
    print("\n  [7f] Risk Function Parameter Sensitivity...")
    risk_slope_mid_values = [0.03, 0.06, 0.09, 0.12]   # baseline slope in mid period
    risk_slope_high_values = [0.10, 0.15, 0.20, 0.25]   # baseline slope in late period

    def risk_function_param(w, slope_mid, slope_high):
        """Parameterized risk function."""
        w_arr = np.asarray(w, dtype=float)
        risk = np.zeros_like(w_arr)
        mask_low = w_arr <= EARLY_WEEK
        mask_mid = (w_arr > EARLY_WEEK) & (w_arr <= MID_WEEK)
        mask_high = w_arr > MID_WEEK
        risk[mask_low] = 0.05
        risk[mask_mid] = 0.05 + slope_mid * (w_arr[mask_mid] - EARLY_WEEK) / (MID_WEEK - EARLY_WEEK)
        risk[mask_high] = 0.05 + slope_mid + slope_high * (w_arr[mask_high] - MID_WEEK) / (40 - MID_WEEK)
        return risk

    risk_sensitivity = []
    for slope_mid in risk_slope_mid_values:
        for slope_high in risk_slope_high_values:
            row_r = {'slope_mid': slope_mid, 'slope_high': slope_high}
            # Recompute optimal timing for each Q2 BMI cluster with this risk function
            for cid in sorted(df_mothers_q2['bmi_cluster'].unique()):
                cdata = df_mothers_q2[df_mothers_q2['bmi_cluster'] == cid]
                crossing = cdata['crossing_week'].dropna()
                if len(crossing) > 0:
                    def obj_r(w):
                        pct = (crossing <= w).mean()
                        if pct < 0.5:
                            return 1e9 * (0.5 - pct + 1)
                        return risk_function_param(w, slope_mid, slope_high) + 0.1 * (1 - pct)
                    res = minimize_scalar(obj_r, bounds=(10, 35), method='bounded')
                    row_r[f'G{cid+1}_opt'] = res.x
                else:
                    row_r[f'G{cid+1}_opt'] = np.nan
            risk_sensitivity.append(row_r)

    df_risk_sens = pd.DataFrame(risk_sensitivity)
    # Show how much optimal week shifts from baseline (slope_mid=0.06, slope_high=0.15)
    baseline_row = df_risk_sens[(df_risk_sens['slope_mid'] == 0.06) &
                                 (df_risk_sens['slope_high'] == 0.15)]
    if len(baseline_row) > 0:
        print(f"    Baseline (mid_slope=0.06, high_slope=0.15) reference weeks:")
        for cid in range(4):
            col = f'G{cid+1}_opt'
            if col in baseline_row.columns:
                print(f"      Group {cid+1}: {baseline_row[col].values[0]:.1f}w")
        # Max shift across all parameter combos
        max_shifts = {}
        for cid in range(4):
            col = f'G{cid+1}_opt'
            if col in df_risk_sens.columns:
                vals = df_risk_sens[col].dropna()
                if len(vals) > 0 and len(baseline_row) > 0:
                    shift = np.max(np.abs(vals - baseline_row[col].values[0]))
                    max_shifts[cid+1] = shift
        print(f"    Max |shift| across all parameter combos: "
              + ", ".join([f"G{k}={v:.2f}w" for k, v in max_shifts.items()]))
    sa_results['risk_param_sensitivity'] = df_risk_sens

    # ---- Generate Sensitivity Analysis Figures ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # (a) Threshold sensitivity — optimal week vs threshold
    ax = axes[0, 0]
    for cid in range(4):
        col = f'cluster_{cid+1}_opt_week'
        if col in df_thresh_sens.columns:
            ax.plot(df_thresh_sens['threshold'] * 100, df_thresh_sens[col],
                    'o-', markersize=6, label=f'Q2 Group {cid+1}')
    ax.axvline(x=Y_THRESHOLD * 100, color='red', linestyle='--', label=f'Default ({Y_THRESHOLD*100:.0f}%)')
    ax.set_xlabel('Y Concentration Threshold (%)')
    ax.set_ylabel('Optimal Week')
    ax.set_title('(a) Threshold Sensitivity: Optimal Week', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Noise → crossing week shift distribution
    ax = axes[0, 1]
    colors_noise = plt.cm.Reds(np.linspace(0.3, 0.9, len(noise_levels) - 1))
    for i, sigma in enumerate([s for s in noise_levels if s > 0]):
        # Sample shifts for this noise level
        shifts_sample = []
        for _ in range(500):
            shift_sum = []
            for mid, grp in data_q2.groupby('mother_id'):
                if len(grp) < 2:
                    continue
                grp_s = grp.sort_values('gestational_weeks')
                weeks_s = grp_s['gestational_weeks'].values
                yv = grp_s['y_conc'].values.copy()
                yv_n = yv + rng.normal(0, sigma, size=len(yv))
                yv_n = np.maximum(yv_n, 0.0)
                try:
                    f1 = interp1d(weeks_s, yv, kind='linear', bounds_error=False,
                                  fill_value=(yv[0], yv[-1]))
                    f2 = interp1d(weeks_s, yv_n, kind='linear', bounds_error=False,
                                  fill_value=(yv_n[0], yv_n[-1]))
                    tw = np.linspace(weeks_s.min(), min(weeks_s.max() + 10, 40), 200)
                    a1 = f1(tw) >= Y_THRESHOLD
                    a2 = f2(tw) >= Y_THRESHOLD
                    if a1.any() and a2.any():
                        shift_sum.append(tw[a2][0] - tw[a1][0])
                except Exception:
                    pass
            if shift_sum:
                shifts_sample.append(np.mean(shift_sum))
        if shifts_sample:
            ax.hist(shifts_sample, bins=40, alpha=0.5, label=f'σ={sigma:.3f}',
                    color=colors_noise[i])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Crossing Week Shift (weeks)')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Noise-Induced Crossing Week Shift', fontweight='bold')
    ax.legend(fontsize=7)

    # (c) Bootstrap CI for Q2 optimal weeks
    ax = axes[0, 2]
    groups_q2 = sorted(q2_bootstrap.keys())
    means_q2 = [q2_bootstrap[g]['mean'] for g in groups_q2]
    ci_lows = [q2_bootstrap[g]['mean'] - q2_bootstrap[g]['ci_95_low'] for g in groups_q2]
    ci_highs = [q2_bootstrap[g]['ci_95_high'] - q2_bootstrap[g]['mean'] for g in groups_q2]
    ax.errorbar(groups_q2, means_q2, yerr=[ci_lows, ci_highs],
                fmt='o', capsize=8, markersize=10, linewidth=2, color='steelblue',
                label='Optimal Week ± 95% CI')
    ax.fill_between([g - 0.3 for g in groups_q2], [q2_bootstrap[g]['ci_90_low'] for g in groups_q2],
                    [q2_bootstrap[g]['ci_90_high'] for g in groups_q2],
                    alpha=0.2, color='steelblue', label='90% CI')
    ax.set_xlabel('BMI Group')
    ax.set_ylabel('Optimal Gestational Week')
    ax.set_title('(c) Q2 Bootstrap CI (n=1000)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Q3 cluster stability — ARI vs perturbation
    ax = axes[1, 0]
    ax.errorbar(df_pert['perturbation_factor'], df_pert['mean_ari'],
                yerr=df_pert['std_ari'], fmt='o-', capsize=5, markersize=8,
                linewidth=2, color='darkorange')
    ax.axhline(y=0.8, color='red', linestyle='--', label='ARI = 0.8 (threshold)')
    ax.fill_between(df_pert['perturbation_factor'],
                    df_pert['mean_ari'] - df_pert['std_ari'],
                    df_pert['mean_ari'] + df_pert['std_ari'],
                    alpha=0.2, color='darkorange')
    ax.set_xlabel('Feature Perturbation Factor (σ)')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('(d) Q3 Cluster Stability Under Noise', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Q4 classifier F1 degradation under noise
    ax = axes[1, 1]
    ref_f1 = df_q4_robust['f1_mean'].iloc[0]
    ax.errorbar(df_q4_robust['noise_sigma'][1:], df_q4_robust['f1_mean'][1:],
                yerr=df_q4_robust['f1_std'][1:], fmt='o-', capsize=5,
                markersize=8, linewidth=2, color='crimson', label='F1 Score')
    ax.axhline(y=ref_f1, color='gray', linestyle=':', label=f'Reference F1={ref_f1:.4f}')
    # Add AUC on twin axis
    ax2 = ax.twinx()
    ax2.plot(df_q4_robust['noise_sigma'][1:], df_q4_robust['auc_mean'][1:],
             's--', markersize=6, color='steelblue', label='AUC')
    ax2.set_ylabel('ROC-AUC', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('F1 Score', color='crimson')
    ax.tick_params(axis='y', labelcolor='crimson')
    ax.set_title('(e) Q4 Classifier Robustness', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # (f) Risk function parameter heatmap — max optimal week shift
    ax = axes[1, 2]
    heatmap_data = df_risk_sens.pivot_table(
        values='G1_opt', index='slope_mid', columns='slope_high', aggfunc='mean')
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax.set_xticks(range(len(risk_slope_high_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in risk_slope_high_values])
    ax.set_yticks(range(len(risk_slope_mid_values)))
    ax.set_yticklabels([f'{v:.2f}' for v in risk_slope_mid_values])
    ax.set_xlabel('Late-Period Slope')
    ax.set_ylabel('Mid-Period Slope')
    ax.set_title('(f) Risk Param Sensitivity: G1 Optimal Wk', fontweight='bold')
    # Annotate cells
    for i in range(len(risk_slope_mid_values)):
        for j in range(len(risk_slope_high_values)):
            val = heatmap_data.values[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val < 15 or val > 25 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Optimal Week')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'Q7_sensitivity_analysis.png', dpi=300)
    plt.close()
    print(f"\n  Figure saved: Q7_sensitivity_analysis.png")

    # ---- Additional figure: Noise Robustness Summary ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # (g) RMSE of crossing week shift vs noise level
    ax = axes2[0]
    ax.plot(df_noise['noise_sigma'], df_noise['rmse_shift_weeks'], 'o-',
            color='darkred', markersize=8, linewidth=2)
    ax.fill_between(df_noise['noise_sigma'],
                    df_noise['ci_95_low'], df_noise['ci_95_high'],
                    alpha=0.2, color='darkred')
    ax.set_xlabel('Noise Standard Deviation (σ)')
    ax.set_ylabel('RMSE of Crossing Week Shift (weeks)')
    ax.set_title('(g) RMSE of Time-Point Shift vs Noise Level', fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Add reference lines
    ax.axhline(y=0.5, color='orange', linestyle='--', label='0.5 week threshold')
    ax.axhline(y=1.0, color='red', linestyle='--', label='1.0 week threshold')
    ax.legend(fontsize=9)

    # (h) Percentage of mothers with large shifts
    ax = axes2[1]
    ax.plot(df_noise['noise_sigma'], df_noise['pct_shift_gt_0.5w'], 's-',
            color='steelblue', markersize=8, linewidth=2, label='|shift| > 0.5w')
    ax.plot(df_noise['noise_sigma'], df_noise['pct_shift_gt_1.0w'], 'D-',
            color='crimson', markersize=8, linewidth=2, label='|shift| > 1.0w')
    ax.set_xlabel('Noise Standard Deviation (σ)')
    ax.set_ylabel('Percentage of Mothers (%)')
    ax.set_title('(h) Proportion with Large Crossing Week Shift', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(FIGURE_DIR / 'Q7_noise_robustness_detail.png', dpi=300)
    plt.close()
    print(f"  Figure saved: Q7_noise_robustness_detail.png")

    # Store raw data for later export
    sa_results['noise_levels_tested'] = noise_levels
    sa_results['threshold_values'] = threshold_values
    sa_results['pert_factors'] = pert_factors
    sa_results['noise_levels_q4'] = noise_levels_q4

    return sa_results


# ============================================================================
# 8. RESULTS OUTPUT
# ============================================================================
def save_results(q1_results, q2_results, q3_results, q4_results, sa_results=None):
    """Save all results to CSV files and generate summary report."""
    print("\n" + "=" * 60)
    print("8. RESULTS OUTPUT")
    print("=" * 60)

    # ---- Q1 results ----
    q1_export = {
        'Metric': ['OLS R²', 'OLS Adj R²', 'Poly R²', 'Poly Adj R²',
                   'CV R² (mean)', 'CV R² (std)', 'CV RMSE (mean)',
                   'Pearson r (age)', 'Pearson p (age)',
                   'Pearson r (bmi)', 'Pearson p (bmi)',
                   'Spearman r (age)', 'Spearman r (bmi)'],
        'Value': [
            q1_results['ols_r2'], q1_results['ols_adj_r2'],
            q1_results['poly_r2'], q1_results['poly_adj_r2'],
            q1_results['cv_r2_mean'], q1_results['cv_r2_std'],
            q1_results['cv_rmse_mean'],
            q1_results['pearson']['age'][0], q1_results['pearson']['age'][1],
            q1_results['pearson']['bmi'][0], q1_results['pearson']['bmi'][1],
            q1_results['spearman']['age'][0], q1_results['spearman']['bmi'][0],
        ]
    }
    pd.DataFrame(q1_export).to_csv(TABLE_DIR / 'Q1_results.csv', index=False)
    print("  → results/tables/Q1_results.csv")

    # ---- Q2 results ----
    q2_results['cluster_summary'].to_csv(TABLE_DIR / 'Q2_BMI_clusters.csv', index=False)
    print("  → results/tables/Q2_BMI_clusters.csv")

    # ---- Q3 results ----
    q3_results['cluster_summary'].to_csv(TABLE_DIR / 'Q3_multifactor_clusters.csv', index=False)
    print("  → results/tables/Q3_multifactor_clusters.csv")

    # ---- Q4 results ----
    q4_results['model_comparison'].to_csv(TABLE_DIR / 'Q4_model_comparison.csv', index=False)
    print("  → results/tables/Q4_model_comparison.csv")

    # ---- Sensitivity Analysis results ----
    if sa_results is not None:
        if 'threshold_sensitivity' in sa_results:
            sa_results['threshold_sensitivity'].to_csv(
                TABLE_DIR / 'SA_threshold_sensitivity.csv', index=False)
            print("  → results/tables/SA_threshold_sensitivity.csv")

        if 'noise_stability' in sa_results:
            sa_results['noise_stability'].to_csv(
                TABLE_DIR / 'SA_noise_stability.csv', index=False)
            print("  → results/tables/SA_noise_stability.csv")

        if 'q3_cluster_stability' in sa_results:
            sa_results['q3_cluster_stability'].to_csv(
                TABLE_DIR / 'SA_q3_cluster_stability.csv', index=False)
            print("  → results/tables/SA_q3_cluster_stability.csv")

        if 'q4_noise_robustness' in sa_results:
            sa_results['q4_noise_robustness'].to_csv(
                TABLE_DIR / 'SA_q4_noise_robustness.csv', index=False)
            print("  → results/tables/SA_q4_noise_robustness.csv")

        if 'risk_param_sensitivity' in sa_results:
            sa_results['risk_param_sensitivity'].to_csv(
                TABLE_DIR / 'SA_risk_param_sensitivity.csv', index=False)
            print("  → results/tables/SA_risk_param_sensitivity.csv")

        # Bootstrap CI export
        if 'q2_bootstrap_ci' in sa_results:
            q2_ci_rows = []
            for g, v in sa_results['q2_bootstrap_ci'].items():
                v['group'] = g
                q2_ci_rows.append(v)
            pd.DataFrame(q2_ci_rows).to_csv(
                TABLE_DIR / 'SA_q2_bootstrap_ci.csv', index=False)
            print("  → results/tables/SA_q2_bootstrap_ci.csv")

        if 'q3_bootstrap_ci' in sa_results:
            q3_ci_rows = []
            for g, v in sa_results['q3_bootstrap_ci'].items():
                v['group'] = g
                q3_ci_rows.append(v)
            pd.DataFrame(q3_ci_rows).to_csv(
                TABLE_DIR / 'SA_q3_bootstrap_ci.csv', index=False)
            print("  → results/tables/SA_q3_bootstrap_ci.csv")

    # ---- Comprehensive Summary Report ----
    summary_lines = [
        "=" * 70,
        "NIPT MATHEMATICAL MODELING — COMPREHENSIVE SUMMARY",
        "=" * 70,
        "",
        "Q1: Y CONCENTRATION vs AGE + BMI",
        "-" * 40,
        f"  Pearson r(age) = {q1_results['pearson']['age'][0]:.4f} (p={q1_results['pearson']['age'][1]:.2e})",
        f"  Pearson r(bmi) = {q1_results['pearson']['bmi'][0]:.4f} (p={q1_results['pearson']['bmi'][1]:.2e})",
        f"  OLS R² = {q1_results['ols_r2']:.4f}, Adj R² = {q1_results['ols_adj_r2']:.4f}",
        f"  Polynomial R² = {q1_results['poly_r2']:.4f}",
        f"  5-Fold CV R² = {q1_results['cv_r2_mean']:.4f} ± {q1_results['cv_r2_std']:.4f}",
        f"  Top features (RF): {list(q1_results['feature_importance'].keys())[:3]}",
        "",
        "Q2: BMI GROUPING → OPTIMAL NIPT TIMING",
        "-" * 40,
    ]

    for _, row in q2_results['cluster_summary'].iterrows():
        summary_lines.append(
            f"  Group {int(row['cluster'])}: BMI [{row['bmi_min']:.1f}, {row['bmi_max']:.1f}], "
            f"n={int(row['n_mothers'])}, Optimal={row['optimal_week']:.1f}w, "
            f"Risk={row['risk_at_optimal']:.4f}, Coverage={row['pct_covered']:.1%}")

    summary_lines += [
        "",
        "Q3: MULTI-FACTOR GROUPING → OPTIMAL NIPT TIMING",
        "-" * 40,
    ]
    for _, row in q3_results['cluster_summary'].iterrows():
        summary_lines.append(
            f"  Group {int(row['cluster'])}: Age={row['mean_age']:.1f}, "
            f"BMI={row['mean_bmi']:.1f}, n={int(row['n_mothers'])}, "
            f"Optimal={row['optimal_week']:.1f}w, Risk={row['risk_at_optimal']:.4f}, "
            f"Coverage={row['pct_covered']:.1%}")

    summary_lines += [
        "",
        "Q4: FEMALE FETUS ABNORMALITY CLASSIFICATION",
        "-" * 40,
        f"  Samples: {q4_results['n_normal']} normal, {q4_results['n_abnormal']} aneuploidy",
        f"  Baseline (|Z|>3): F1={q4_results['baseline']['f1']:.4f}",
    ]
    for _, row in q4_results['model_comparison'].iterrows():
        summary_lines.append(
            f"  {row['Model']:<25s}: F1={row['F1-Score']:.4f}, AUC={row['ROC-AUC']:.4f}, "
            f"CV F1={row['CV F1 (mean)']:.4f}±{row['CV F1 (std)']:.4f}")

    # Sensitivity Analysis summary
    if sa_results is not None:
        summary_lines += [
            "",
            "SENSITIVITY ANALYSIS & NOISE ROBUSTNESS",
            "-" * 40,
        ]
        # Q2 Bootstrap CI
        if 'q2_bootstrap_ci' in sa_results:
            summary_lines.append("  Q2 Optimal Timing Bootstrap 95% CI:")
            for g, v in sa_results['q2_bootstrap_ci'].items():
                summary_lines.append(
                    f"    Group {g}: {v['mean']:.1f}w [{v['ci_95_low']:.1f}, {v['ci_95_high']:.1f}]")
        # Q3 Bootstrap CI
        if 'q3_bootstrap_ci' in sa_results:
            summary_lines.append("  Q3 Optimal Timing Bootstrap 95% CI:")
            for g, v in sa_results['q3_bootstrap_ci'].items():
                summary_lines.append(
                    f"    Group {g}: {v['mean']:.1f}w [{v['ci_95_low']:.1f}, {v['ci_95_high']:.1f}]")
        # Noise stability
        if 'noise_stability' in sa_results:
            df_ns = sa_results['noise_stability']
            worst = df_ns[df_ns['noise_sigma'] == df_ns['noise_sigma'].max()]
            if len(worst) > 0:
                summary_lines.append(
                    f"  Max noise (σ={worst['noise_sigma'].values[0]:.3f}): "
                    f"mean shift={worst['mean_shift_weeks'].values[0]:+.3f}w, "
                    f"RMSE={worst['rmse_shift_weeks'].values[0]:.3f}w")
        # Q4 noise robustness
        if 'q4_noise_robustness' in sa_results:
            df_q4r = sa_results['q4_noise_robustness']
            ref_f1 = df_q4r['f1_mean'].iloc[0]
            mod_noise = df_q4r[df_q4r['noise_sigma'] == 0.20]
            if len(mod_noise) > 0:
                summary_lines.append(
                    f"  Q4 Classifier at σ=0.20: F1={mod_noise['f1_mean'].values[0]:.4f} "
                    f"(drop={ref_f1 - mod_noise['f1_mean'].values[0]:.4f} from baseline {ref_f1:.4f})")

    summary_lines += [
        "",
        "FIGURES GENERATED",
        "-" * 40,
        "  results/figures/Q1_analysis.png",
        "  results/figures/Q2_BMI_optimization.png",
        "  results/figures/Q3_multifactor_optimization.png",
        "  results/figures/Q4_female_abnormality.png",
    ]
    if sa_results is not None:
        summary_lines += [
            "  results/figures/Q7_sensitivity_analysis.png",
            "  results/figures/Q7_noise_robustness_detail.png",
        ]
    summary_lines += [
        "",
        "=" * 70,
    ]

    summary_text = '\n'.join(summary_lines)
    with open(OUTPUT_DIR / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print("  → results/summary_report.txt")
    print("\n" + summary_text)


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Execute the complete NIPT modeling pipeline."""
    print("\n" + "█" * 60)
    print("█  NIPT MATHEMATICAL MODELING — COMPLETE PIPELINE")
    print("█  2025 MCM/ICM Problem C")
    print("█" * 60)

    # 1. Load
    df_male, df_female = load_data()

    # 2. Preprocess
    df_male, df_female = preprocess_data(df_male, df_female)

    # 3. Q1
    q1_results, _ = solve_q1(df_male)

    # 4. Q2
    q2_results = solve_q2(df_male)

    # 5. Q3
    q3_results = solve_q3(df_male)

    # 6. Q4
    q4_results = solve_q4(df_female)

    # 7. Sensitivity Analysis & Noise Robustness
    sa_results = sensitivity_analysis(df_male, df_female,
                                       q2_results, q3_results, q4_results)

    # 8. Output
    save_results(q1_results, q2_results, q3_results, q4_results, sa_results)

    print("\n" + "█" * 60)
    print("█  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"  Results directory: {OUTPUT_DIR.resolve()}")
    print(f"  Figures: {FIGURE_DIR.resolve()}")
    print(f"  Tables:  {TABLE_DIR.resolve()}")
    print(f"  Sensitivity figures: Q7_sensitivity_analysis.png, Q7_noise_robustness_detail.png")
    print()


if __name__ == '__main__':
    main()
