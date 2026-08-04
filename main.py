#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT Mathematical Modeling — Main Entry Point
==============================================
2025 CUMCM Problem C: NIPT Timing Selection & Fetal Abnormality Detection

Pipeline:
  1. Data loading & preprocessing
  2. Problem 1 — Y Concentration vs Age + BMI relationship model
  3. Problem 2 — BMI-based grouping → optimal NIPT timing
  4. Problem 3 — Multi-factor extended model
  5. Problem 4 — Female fetus abnormality classification
  6. Sensitivity analysis & noise robustness
  7. Results aggregation & export

Usage:
    python main.py

All outputs are saved to results/figures/ and results/tables/.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, FIGURE_DIR, TABLE_DIR, RANDOM_SEED,
    Y_THRESHOLD, EARLY_WEEK, MID_WEEK,
)

# ---- Preprocessing ----
from preprocessing.data_process import preprocess

# ---- Models ----
from models.problem1_model import fit_problem1
from models.problem2_model import fit_problem2
from models.problem3_model import fit_problem3
from models.problem4_model import fit_problem4

# ---- Evaluation ----
from evaluation.metrics import (
    bootstrap_optimal_week,
    noise_robustness_crossing,
    threshold_sensitivity,
    q4_noise_robustness,
    q3_cluster_stability,
)

# ---- Visualization ----
from visualization.plot import (
    plot_eda,
    plot_problem1,
    plot_problem2,
    plot_problem3,
    plot_problem4,
    plot_sensitivity,
)


# ============================================================================
# Results Export
# ============================================================================
def export_results(p1, p2, p3, p4, sa):
    """Export all results to CSV tables and generate summary report."""
    print("\n" + "=" * 60)
    print("[Export] Saving results to tables and report")
    print("=" * 60)

    # --- Q1 ---
    q1_rows = [
        {'Metric': 'Model 1 R²', 'Value': p1['model1']['r2']},
        {'Metric': 'Model 1 Adj R²', 'Value': p1['model1']['r2_adj']},
        {'Metric': 'Model 1 F-statistic', 'Value': p1['model1']['F_stat']},
        {'Metric': 'Model 1 p(F)', 'Value': p1['model1']['p_F']},
        {'Metric': 'Model 2 R²', 'Value': p1['model2']['r2']},
        {'Metric': 'Model 3 R²', 'Value': p1['model3']['r2']},
        {'Metric': 'CV R² (mean)', 'Value': p1['cv']['r2_mean']},
        {'Metric': 'CV R² (std)', 'Value': p1['cv']['r2_std']},
        {'Metric': 'CV RMSE (mean)', 'Value': p1['cv']['rmse_mean']},
        {'Metric': 'Best Model', 'Value': p1['best_model']},
        {'Metric': 'Best R²', 'Value': p1['best_r2']},
    ]
    # Add correlations
    for var in ['weeks_num', 'bmi', 'age']:
        q1_rows.append({
            'Metric': f'Pearson r ({var})',
            'Value': p1['correlations']['pearson'][var]['r'],
        })
        q1_rows.append({
            'Metric': f'Spearman ρ ({var})',
            'Value': p1['correlations']['spearman'][var]['rho'],
        })
    pd.DataFrame(q1_rows).to_csv(TABLE_DIR / 'Q1_results.csv', index=False)
    print("  → Q1_results.csv")

    # --- Q2 ---
    p2['group_stats'].to_csv(TABLE_DIR / 'Q2_BMI_groups.csv', index=False)
    p2['cluster_results'].to_csv(TABLE_DIR / 'Q2_KMeans_clusters.csv', index=False)
    print("  → Q2_BMI_groups.csv, Q2_KMeans_clusters.csv")

    # --- Q3 ---
    p3['g3_summary'].to_csv(TABLE_DIR / 'Q3_multifactor_groups.csv', index=False)
    p3['cluster_results'].to_csv(TABLE_DIR / 'Q3_KMeans_clusters.csv', index=False)
    if p3['q2_q3_comparison'] is not None:
        p3['q2_q3_comparison'].to_csv(TABLE_DIR / 'Q3_Q2_vs_Q3_comparison.csv', index=False)
    print("  → Q3_multifactor_groups.csv, Q3_KMeans_clusters.csv")

    # --- Q4 ---
    q4_rows = [
        {'Metric': 'Baseline Z-threshold Accuracy', 'Value': p4['baseline_z']['accuracy']},
        {'Metric': 'Baseline Z-threshold F1', 'Value': p4['baseline_z']['f1']},
        {'Metric': 'LogReg Accuracy', 'Value': p4['logreg_metrics']['accuracy']},
        {'Metric': 'LogReg Precision', 'Value': p4['logreg_metrics']['precision']},
        {'Metric': 'LogReg Recall', 'Value': p4['logreg_metrics']['recall']},
        {'Metric': 'LogReg F1', 'Value': p4['logreg_metrics']['f1']},
        {'Metric': 'LogReg AUC', 'Value': p4['logreg_metrics']['auc']},
        {'Metric': 'LogReg CV AUC (mean)', 'Value': p4['logreg_metrics']['cv_auc_mean']},
        {'Metric': 'RF Accuracy', 'Value': p4['rf_metrics']['accuracy']},
        {'Metric': 'RF F1', 'Value': p4['rf_metrics']['f1']},
        {'Metric': 'RF AUC', 'Value': p4['rf_metrics']['auc']},
        {'Metric': 'Normal samples', 'Value': p4['n_normal']},
        {'Metric': 'Abnormal samples', 'Value': p4['n_abnormal']},
    ]
    pd.DataFrame(q4_rows).to_csv(TABLE_DIR / 'Q4_results.csv', index=False)
    p4['logreg_coefficients'].to_csv(TABLE_DIR / 'Q4_logreg_coefficients.csv', index=False)
    print("  → Q4_results.csv, Q4_logreg_coefficients.csv")

    # --- Sensitivity ---
    if sa:
        for key, fname in [
            ('threshold_sensitivity', 'SA_threshold.csv'),
            ('noise_stability', 'SA_noise_stability.csv'),
            ('q3_cluster_stability', 'SA_q3_stability.csv'),
            ('q4_noise_robustness', 'SA_q4_robustness.csv'),
            ('risk_param_sensitivity', 'SA_risk_params.csv'),
        ]:
            if key in sa and sa[key] is not None:
                sa[key].to_csv(TABLE_DIR / fname, index=False)
                print(f"  → {fname}")

    # --- Summary Report ---
    lines = [
        "=" * 70,
        "NIPT MATHEMATICAL MODELING — COMPREHENSIVE SUMMARY",
        "2025 CUMCM Problem C",
        "=" * 70,
        "",
        "PROBLEM 1: Y CONCENTRATION ~ AGE + BMI",
        "-" * 40,
        f"  Best Model: {p1['best_model']} (R²={p1['best_r2']:.4f})",
        f"  Model 1: R²={p1['model1']['r2']:.4f}, F={p1['model1']['F_stat']:.1f} (p={p1['model1']['p_F']:.2e})",
        f"  CV R² = {p1['cv']['r2_mean']:.4f} ± {p1['cv']['r2_std']:.4f}",
        f"  Spearman: Y~weeks ρ={p1['correlations']['spearman']['weeks_num']['rho']:.3f}, "
        f"Y~BMI ρ={p1['correlations']['spearman']['bmi']['rho']:.3f}",
        "",
        "PROBLEM 2: BMI GROUPING → OPTIMAL NIPT TIMING",
        "-" * 40,
    ]
    for _, r in p2['group_stats'].iterrows():
        if not pd.isna(r['optimal_week']):
            lines.append(
                f"  {r['bmi_group']}: n={int(r['n_mothers'])}, median={r['median_达标_week']:.1f}w, "
                f"optimal={r['optimal_week']:.1f}w [{r['risk_level']}]")

    lines += [
        "",
        "PROBLEM 3: MULTI-FACTOR EXTENDED MODEL",
        "-" * 40,
        f"  LASSO selected {len(p3['lasso_selected_features'])} features, R²={p3['lasso_r2']:.4f}",
        f"  K-Means best k={p3['best_k']} (silhouette={p3['best_silhouette']:.4f})",
    ]
    for _, r in p3['g3_summary'].iterrows():
        lines.append(
            f"  {r['bmi_group']}:达标率={r['达标_rate']:.1%}, optimal={r['optimal_week']:.1f}w")

    lines += [
        "",
        "PROBLEM 4: FEMALE FETUS ABNORMALITY CLASSIFICATION",
        "-" * 40,
        f"  Samples: {p4['n_normal']} normal, {p4['n_abnormal']} abnormal",
        f"  Baseline (|Z|>3): F1={p4['baseline_z']['f1']:.4f}",
        f"  Logistic Regression: F1={p4['logreg_metrics']['f1']:.4f}, AUC={p4['logreg_metrics']['auc']:.4f}",
        f"  Random Forest: F1={p4['rf_metrics']['f1']:.4f}, AUC={p4['rf_metrics']['auc']:.4f}",
    ]
    if p4['ae_validation']:
        lines.append(f"  AE validation (LogReg): Acc={p4['ae_validation']['logreg']['accuracy']:.4f}")

    lines += [
        "",
        "FIGURES GENERATED",
        "-" * 40,
        "  fig_EDA.png",
        "  fig_problem1.png / fig_problem2.png / fig_problem3.png / fig_problem4.png",
        "  fig_sensitivity.png / fig_noise_robustness.png",
        "",
        "=" * 70,
    ]

    report = '\n'.join(lines)
    with open(RESULTS_DIR / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("  → summary_report.txt")
    print("\n" + report)


# ============================================================================
# Sensitivity Analysis Orchestrator
# ============================================================================
def run_sensitivity_analysis(data, data_latest, p2, p3, p4):
    """Run all sensitivity and robustness checks."""
    print("\n" + "=" * 60)
    print("[Sensitivity Analysis] Noise robustness & parameter sensitivity")
    print("=" * 60)

    sa = {}

    # --- Threshold Sensitivity ---
    print("\n  [SA-1] Y-threshold sensitivity...")
    sa['threshold_sensitivity'] = threshold_sensitivity(
        p2['earliest'], p2['risk_function'])

    # --- Noise → Crossing Week Stability ---
    print("  [SA-2] Noise injection → crossing week stability...")
    sa['noise_stability'] = noise_robustness_crossing(data)

    # --- Bootstrap CI for Q2 ---
    print("  [SA-3] Bootstrap CI for Q2 optimal weeks...")
    q2_bootstrap = {}
    for grp_name, grp_data in p2['earliest'].groupby('bmi_group', observed=False):
        crossing = grp_data['earliest_达标_weeks'].dropna().values
        ci = bootstrap_optimal_week(crossing, p2['risk_function'])
        if ci:
            q2_bootstrap[grp_name] = ci
    sa['q2_bootstrap_ci'] = q2_bootstrap

    # --- Bootstrap CI for Q3 ---
    print("  [SA-4] Bootstrap CI for Q3 optimal weeks...")
    q3_bootstrap = {}
    df_q3 = p3['df_with_pred']
    for cid in sorted(df_q3['cluster'].unique()):
        crossing = df_q3[df_q3['cluster'] == cid]['pred_达标'].dropna().values
        ci = bootstrap_optimal_week(crossing, p2['risk_function'])
        if ci:
            q3_bootstrap[cid] = ci
    sa['q3_bootstrap_ci'] = q3_bootstrap

    # --- Q3 Cluster Stability ---
    print("  [SA-5] Q3 cluster stability under feature perturbation...")
    sa['q3_cluster_stability'] = q3_cluster_stability(
        df_q3, ['age', 'height', 'weight', 'bmi'], p3['best_k'])

    # --- Q4 Classifier Robustness ---
    print("  [SA-6] Q4 classifier robustness to Z-score noise...")
    sa['q4_noise_robustness'] = q4_noise_robustness(data_latest)

    # --- Risk Parameter Sensitivity ---
    print("  [SA-7] Risk function parameter sensitivity...")
    from models.problem2_model import risk_function
    risk_rows = []
    for slope_mid in [0.03, 0.06, 0.09, 0.12]:
        for slope_high in [0.10, 0.15, 0.20, 0.25]:
            row = {'slope_mid': slope_mid, 'slope_high': slope_high}

            def param_risk_fn(w):
                return _param_risk(w, slope_mid, slope_high)

            for grp_name, grp_data in p2['earliest'].groupby('bmi_group', observed=False):
                crossing = grp_data['earliest_达标_weeks'].dropna().values
                if len(crossing) > 0:
                    ci = bootstrap_optimal_week(crossing, param_risk_fn, n_bootstrap=200)
                    row[f'{grp_name}_opt'] = ci['mean'] if ci else np.nan
                else:
                    row[f'{grp_name}_opt'] = np.nan
            risk_rows.append(row)
    sa['risk_param_sensitivity'] = pd.DataFrame(risk_rows)

    print("[Sensitivity Analysis] Complete.\n")
    return sa


def _param_risk(w, slope_mid, slope_high):
    """Parameterized risk function for sensitivity analysis."""
    w_arr = np.asarray(w, dtype=float)
    risk = np.zeros_like(w_arr)
    mask_low = w_arr <= EARLY_WEEK
    mask_mid = (w_arr > EARLY_WEEK) & (w_arr <= MID_WEEK)
    mask_high = w_arr > MID_WEEK
    risk[mask_low] = 0.05
    risk[mask_mid] = 0.05 + slope_mid * (w_arr[mask_mid] - EARLY_WEEK) / (MID_WEEK - EARLY_WEEK)
    risk[mask_high] = 0.05 + slope_mid + slope_high * (w_arr[mask_high] - MID_WEEK) / (40 - MID_WEEK)
    return risk


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Execute the complete NIPT modeling pipeline."""
    print("\n" + "█" * 60)
    print("█  NIPT MATHEMATICAL MODELING — COMPLETE PIPELINE")
    print("█  2025 CUMCM Problem C")
    print("█" * 60)
    print(f"█  Random Seed: {RANDOM_SEED}")
    print(f"█  Y Threshold: {Y_THRESHOLD}%")
    print(f"█  Output: {RESULTS_DIR.resolve()}")
    print("█" * 60)

    # ========================================================================
    # Step 1: Data Loading & Preprocessing
    # ========================================================================
    data, data_latest = preprocess()

    # ========================================================================
    # Step 2: Problem 1 — Y Concentration vs Age + BMI
    # ========================================================================
    p1_results = fit_problem1(data_latest)

    # ========================================================================
    # Step 3: Problem 2 — BMI Grouping → Optimal NIPT Timing
    # ========================================================================
    p2_results = fit_problem2(data, data_latest)

    # ========================================================================
    # Step 4: Problem 3 — Multi-Factor Extended Model
    # ========================================================================
    p3_results = fit_problem3(data_latest, p2_results)

    # ========================================================================
    # Step 5: Problem 4 — Female Fetus Abnormality Classification
    # ========================================================================
    p4_results = fit_problem4(data_latest)

    # ========================================================================
    # Step 6: Sensitivity Analysis & Noise Robustness
    # ========================================================================
    sa_results = run_sensitivity_analysis(data, data_latest,
                                           p2_results, p3_results, p4_results)

    # ========================================================================
    # Step 7: Generate All Figures
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Visualization] Generating publication figures")
    print("=" * 60)
    plot_eda(data_latest)
    plot_problem1(p1_results)
    plot_problem2(p2_results)
    plot_problem3(p3_results)
    plot_problem4(p4_results)
    plot_sensitivity(sa_results)
    print("[Visualization] All figures generated.\n")

    # ========================================================================
    # Step 8: Export Results
    # ========================================================================
    export_results(p1_results, p2_results, p3_results, p4_results, sa_results)

    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "█" * 60)
    print("█  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"  Results:  {RESULTS_DIR.resolve()}")
    print(f"  Figures:  {FIGURE_DIR.resolve()}")
    print(f"  Tables:   {TABLE_DIR.resolve()}")
    print(f"  Report:   {RESULTS_DIR / 'summary_report.txt'}")
    print("█" * 60 + "\n")


if __name__ == '__main__':
    main()
