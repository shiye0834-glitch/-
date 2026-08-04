# -*- coding: utf-8 -*-
"""
NIPT Mathematical Modeling — Global Configuration
==================================================
2025 CUMCM Problem C: NIPT Timing Selection & Fetal Abnormality Detection

All tunable parameters, paths, constants, and figure settings live here.
Modify this file to adapt the pipeline to a different data source or
experimental setting without touching model code.
"""

import numpy as np
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT.parent / 'workspace' / 'data_raw' / '附件.xlsx'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURE_DIR = RESULTS_DIR / 'figures'
TABLE_DIR = RESULTS_DIR / 'tables'

# Ensure output directories exist
for d in [RESULTS_DIR, FIGURE_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Random Seed
# ============================================================================
RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)

# ============================================================================
# Physical / Clinical Constants
# ============================================================================
Y_THRESHOLD = 4.0          # Y染色体浓度达标阈值 (%)
Y_THRESHOLD_DECIMAL = 0.04 # 同上，以小数表示
GC_LOW = 0.40              # GC含量正常范围下界
GC_HIGH = 0.60             # GC含量正常范围上界
Z_THRESHOLD = 3.0          # |Z| > 3 → 潜在非整倍体

# ============================================================================
# Gestational Risk Hierarchy
# ============================================================================
EARLY_WEEK = 12            # ≤12周：低风险
MID_WEEK = 27              # 13–27周：高风险
LATE_WEEK = 28             # ≥28周：极高风险

# ============================================================================
# Risk Function Parameters (for Q2/Q3 optimization)
# ============================================================================
RISK_BASELINE = 0.05       # ≤12周基线风险
RISK_SLOPE_MID = 0.06      # 13-27周风险增量
RISK_SLOPE_HIGH = 0.15     # ≥28周风险增量
RISK_COVERAGE_PENALTY = 0.1  # 覆盖率不足惩罚系数

# ============================================================================
# BMI Grouping
# ============================================================================
BMI_BINS = [20, 28, 32, 36, 40, 60]
BMI_LABELS = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '≥40']

# ============================================================================
# Column Position Mapping (iloc-based for encoding robustness)
# ============================================================================
COL_POS = {
    'sample_id':          0,   # A
    'patient_code':       1,   # B
    'age':                2,   # C
    'height':             3,   # D
    'weight':             4,   # E
    'last_period':        5,   # F
    'ivf':                6,   # G
    'test_date':          7,   # H
    'blood_draw_count':   8,   # I
    'gestational_weeks':  9,   # J
    'bmi':                10,  # K
    'total_reads':        11,  # L
    'align_ratio':        12,  # M
    'dup_ratio':          13,  # N
    'unique_reads':       14,  # O
    'gc_content':         15,  # P
    'z_chr13':            16,  # Q
    'z_chr18':            17,  # R
    'z_chr21':            18,  # S
    'z_chrX':             19,  # T
    'z_chrY':             20,  # U
    'y_conc':             21,  # V  (比例，非百分比)
    'x_conc':             22,  # W
    'gc_chr13':           23,  # X
    'gc_chr18':           24,  # Y
    'gc_chr21':           25,  # Z
    'filtered_ratio':     26,  # AA
    'aneuploidy':         27,  # AB
    'pregnancy_count':    28,  # AC
    'delivery_count':     29,  # AD
    'fetal_health':       30,  # AE
}

# ============================================================================
# Feature Sets (pre-defined for each subquestion to avoid ad-hoc lists)
# ============================================================================
Q1_FEATURES = ['weeks_num', 'bmi', 'age']
Q3_CANDIDATE_FEATURES = ['weeks_num', 'bmi', 'age', 'height', 'weight', 'x_conc']
Q4_FEATURES = [
    'z_chr13', 'z_chr18', 'z_chr21', 'z_chrX',
    'x_conc', 'gc_chr13', 'gc_chr18', 'gc_chr21', 'gc_content',
    'filtered_ratio', 'align_ratio', 'dup_ratio',
    'age', 'bmi', 'weeks_num'
]
Q4_Z_COLS = ['z_chr13', 'z_chr18', 'z_chr21']

# ============================================================================
# Machine Learning Hyperparameters
# ============================================================================
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
LASSO_CV_FOLDS = 5
LOGREG_MAX_ITER = 2000
N_CLUSTERS_Q2 = 4
SILHOUETTE_K_RANGE = range(2, 7)

# ============================================================================
# Sensitivity Analysis
# ============================================================================
NOISE_LEVELS_Y = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010]
NOISE_LEVELS_Q4 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
THRESHOLD_RANGE = np.arange(0.030, 0.055, 0.005)
N_BOOTSTRAP = 1000
N_NOISE_TRIALS = 200

# ============================================================================
# Matplotlib Global Settings
# ============================================================================
import matplotlib as mpl
mpl.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
