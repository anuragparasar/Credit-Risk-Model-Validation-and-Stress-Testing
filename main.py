import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ==========================================
# 1. LENDINGCLUB DATA INGESTION & CHRONOLOGICAL SPLITTING
# ==========================================
def load_and_prep_lendingclub(filepath='loan.csv'):
    """
    Loads and preprocesses real LendingClub data.
    Creates a true chronological Out-of-Time (OOT) split based on issue date.
    """
    print(f"Loading LendingClub dataset from {filepath}...")
    
    # Standard features found in the Kaggle LendingClub dataset
    usecols = ['issue_d', 'loan_status', 'dti', 'fico_range_low', 
               'annual_inc', 'revol_util', 'loan_amnt', 'int_rate']
    
    try:
        df = pd.read_csv(filepath, usecols=usecols)
    except FileNotFoundError:
        raise FileNotFoundError(f"Please download the LendingClub dataset from Kaggle and save it as '{filepath}' in this directory.")

    # 1. Define Target Variable (Binary: 1 = Default/Charged Off, 0 = Fully Paid)
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
    
    # 2. Clean numeric columns (Kaggle sometimes stores percentages as strings)
    for col in ['revol_util', 'int_rate']:
        if df[col].dtype == 'O':
            df[col] = df[col].str.rstrip('%').astype(float)
            
    # 3. Time-based sorting for genuine OOT splitting
    # Convert date strings (e.g., 'Dec-2015') to datetime
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df = df.sort_values('issue_d').reset_index(drop=True)
    
    # Drop rows without an issue date or target
    df = df.dropna(subset=['issue_d', 'default'])
    
    # 4. Temporal Split: 80% In-Time (Older Loans), 20% Out-of-Time (Newer Loans)
    # This perfectly simulates how banks test models against future macroeconomic conditions
    split_idx = int(len(df) * 0.8)
    df_in_time = df.iloc[:split_idx].copy()
    df_oot = df.iloc[split_idx:].copy()
    
    features = ['dti', 'fico_range_low', 'annual_inc', 'revol_util']
    
    print(f"In-Time Period:  {df_in_time['issue_d'].min().date()} to {df_in_time['issue_d'].max().date()}")
    print(f"Out-of-Time Period: {df_oot['issue_d'].min().date()} to {df_oot['issue_d'].max().date()}")
    
    return df_in_time, df_oot, features

# ==========================================
# 2. VALIDATION METRICS & STABILITY (PSI/KS)
# ==========================================
def calculate_psi(expected, actual, bins=10):
    """
    Calculates Population Stability Index (PSI) using quantile-based deciles.
    This is the mathematically correct industry standard for credit scores.
    """
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / len(actual))

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)

def calculate_ks_statistic(y_true, y_prob):
    """Calculates Kolmogorov-Smirnov (KS) statistic between goods and bads."""
    ks_stat, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return ks_stat

def evaluate_model(y_true, y_prob):
    """Evaluates discrimination and calibration metrics."""
    auc = roc_auc_score(y_true, y_prob)
    return {
        'AUC': round(auc, 4),
        'Gini': round(2 * auc - 1, 4),
        'KS_Stat': round(calculate_ks_statistic(y_true, y_prob), 4),
        'Log_Loss': round(log_loss(y_true, y_prob), 4),
        'Brier': round(brier_score_loss(y_true, y_prob), 4)
    }

# ==========================================
# 3. MODEL DEVELOPMENT & CHALLENGER AUDIT
# ==========================================
if __name__ == "__main__":
    print("--- 1. Data Extraction & Chronological Splitting ---")
    
    # If using the downloaded Kaggle file directly, you may need to change 'loan.csv' 
    # to 'accepted_2007_to_2018Q4.csv' or 'accepted_2007_to_2018Q4.csv.gz'
    df_in_time, df_oot, features = load_and_prep_lendingclub('loan.csv')
    
    # 75/25 Train vs Out-of-Sample (OOS) validation split on the In-Time data
    X_train, X_oos, y_train, y_oos = train_test_split(
        df_in_time[features], df_in_time['default'], 
        test_size=0.25, random_state=42
    )
    X_oot, y_oot = df_oot[features], df_oot['default']
    
    print(f"\nSample Sizes -> Train: {len(X_train):,}, OOS: {len(X_oos):,}, OOT: {len(X_oot):,}")
    
    print("\n--- 2. Baseline Model: Logistic Regression ---")
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    
    lr_prob_train = baseline_pipeline.predict_proba(X_train)[:, 1]
    lr_prob_oos = baseline_pipeline.predict_proba(X_oos)[:, 1]
    lr_prob_oot = baseline_pipeline.predict_proba(X_oot)[:, 1]
    
    print("Baseline OOS Metrics:", evaluate_model(y_oos, lr_prob_oos))
    print("Baseline OOT Metrics:", evaluate_model(y_oot, lr_prob_oot))
    
    print("\n--- 3. Challenger Model: LightGBM ---")
    challenger_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbosity=-1))
    ])
    challenger_pipeline.fit(X_train, y_train)
    
    lgb_prob_train = challenger_pipeline.predict_proba(X_train)[:, 1]
    lgb_prob_oos = challenger_pipeline.predict_proba(X_oos)[:, 1]
    lgb_prob_oot = challenger_pipeline.predict_proba(X_oot)[:, 1]
    
    print("Challenger OOS Metrics:", evaluate_model(y_oos, lgb_prob_oos))
    print("Challenger OOT Metrics:", evaluate_model(y_oot, lgb_prob_oot))
    
    print("\n--- 4. Model Drift & Stability Assessment (PSI) ---")
    psi_baseline = calculate_psi(lr_prob_train, lr_prob_oot)
    psi_challenger = calculate_psi(lgb_prob_train, lgb_prob_oot)
    
    print(f"Baseline Score PSI (Train vs OOT):   {psi_baseline:.4f}")
    print(f"Challenger Score PSI (Train vs OOT): {psi_challenger:.4f}")
    
    # Regulatory PSI thresholds
    for name, psi in [("Baseline", psi_baseline), ("Challenger", psi_challenger)]:
        if psi >= 0.25:
            print(f"CRITICAL [{name}]: Severe score drift (PSI >= 0.25).")
        elif psi >= 0.10:
            print(f"WARNING [{name}]: Moderate score shift (0.10 <= PSI < 0.25).")
        else:
            print(f"STABLE [{name}]: Distribution remains well-aligned (PSI < 0.10).")

   # ==========================================
    # 5. MACROECONOMIC STRESS TEST (SIMULATED RECESSION)
    # ==========================================
    print("\n--- 5. Macroeconomic Stress Test (Simulated Recession) ---")
    
    # Create a copy of the OOT data to apply the shock
    X_oot_shocked = X_oot.copy()
    
    # Simulate a severe stagflation/recession environment:
    X_oot_shocked['annual_inc'] *= 0.85       # Income drops by 15% due to job losses
    X_oot_shocked['dti'] *= 1.176              # DTI spikes by 17.6% 
    X_oot_shocked['fico_range_low'] -= 50     # FICO broadly contracts by 50 points
    X_oot_shocked['revol_util'] *= 1.20       # Revolving utilization spikes by 20%
    
    # Generate predictions on the shocked data
    lr_prob_shocked = baseline_pipeline.predict_proba(X_oot_shocked)[:, 1]
    lgb_prob_shocked = challenger_pipeline.predict_proba(X_oot_shocked)[:, 1]
    
    # Evaluate the models under stress 
    # (Note: Calibration drops because we evaluate shocked X against historical, unshocked y)
    print("Shocked OOT Metrics (Logistic Regression):")
    print(evaluate_model(y_oot, lr_prob_shocked))
    
    print("\nShocked OOT Metrics (LightGBM):")
    print(evaluate_model(y_oot, lgb_prob_shocked))
    
    # Check the structural decay via PSI
    psi_lr_shock = calculate_psi(lr_prob_train, lr_prob_shocked)
    psi_lgb_shock = calculate_psi(lgb_prob_train, lgb_prob_shocked)
    
    print(f"\nStress Test PSI (Logistic Regression): {psi_lr_shock:.4f}")
    print(f"Stress Test PSI (LightGBM): {psi_lgb_shock:.4f}")
