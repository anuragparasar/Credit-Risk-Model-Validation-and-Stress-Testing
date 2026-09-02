import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA INGESTION & CHRONOLOGICAL SPLITTING
# ==========================================
def load_and_prep_lendingclub(filepath='loan.csv'):
    print(f"Loading dataset from '{filepath}'...")
    usecols = ['issue_d', 'loan_status', 'dti', 'fico_range_low', 
               'annual_inc', 'revol_util', 'loan_amnt', 'int_rate']
    
    df = pd.read_csv(filepath, usecols=usecols)

    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    df['default'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)
    
    for col in ['revol_util', 'int_rate']:
        if col in df.columns and df[col].dtype == 'O':
            df[col] = df[col].str.rstrip('%').astype(float)
            
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df = df.sort_values('issue_d').reset_index(drop=True)
    df = df.dropna(subset=['issue_d', 'default'])
    
    split_idx = int(len(df) * 0.8)
    df_in_time = df.iloc[:split_idx].copy()
    df_oot = df.iloc[split_idx:].copy()
    
    features = ['dti', 'fico_range_low', 'annual_inc', 'revol_util']
    return df_in_time, df_oot, features

# ==========================================
# 2. METRICS & DRIFT CALCULATIONS
# ==========================================
def calculate_psi(expected, actual, bins=10):
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges) 
    
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / len(actual))

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def evaluate_model(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    ks_stat, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return {
        'AUC': round(float(auc), 4),
        'KS_Stat': round(float(ks_stat), 4),
        'Brier_Score': round(float(brier_score_loss(y_true, y_prob)), 4),
    }

def print_psi_verdict(name, psi):
    if psi >= 0.25:
        print(f"  CRITICAL [{name}]: PSI = {psi:.4f}")
    elif psi >= 0.10:
        print(f"  WARNING  [{name}]: PSI = {psi:.4f}")
    else:
        print(f"  STABLE   [{name}]: PSI = {psi:.4f}")

# ==========================================
# 3. PIPELINE SETUP & STRESS TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    df_in_time, df_oot, features = load_and_prep_lendingclub('loan.csv')
    
    X_train, X_oos, y_train, y_oos = train_test_split(
        df_in_time[features], df_in_time['default'], test_size=0.25, random_state=42
    )
    X_oot, y_oot = df_oot[features], df_oot['default']
    
    print("\n--- Training Models ---")
    # Reduced to 5 bins to enforce stricter stability under stress
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('binner', KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')),
        ('classifier', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    
    lgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', lgb.LGBMClassifier(n_estimators=60, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=100, random_state=42, verbosity=-1))
    ])
    lgb_pipeline.fit(X_train, y_train)
    
    lr_prob_train = lr_pipeline.predict_proba(X_train)[:, 1]
    lgb_prob_train = lgb_pipeline.predict_proba(X_train)[:, 1]
    
    print("\n--- Simulated Macroeconomic Stress Test ---")
    X_oot_shocked = X_oot.copy()
    X_oot_shocked['annual_inc'] *= 0.90       
    X_oot_shocked['dti'] *= 1.11              
    X_oot_shocked['fico_range_low'] -= 25     
    X_oot_shocked['revol_util'] *= 1.15       
    
    lr_prob_shocked = lr_pipeline.predict_proba(X_oot_shocked)[:, 1]
    lgb_prob_shocked = lgb_pipeline.predict_proba(X_oot_shocked)[:, 1]
    
    print("\nShocked OOT Metrics:")
    print("Logistic Regression:", evaluate_model(y_oot, lr_prob_shocked))
    print("LightGBM:           ", evaluate_model(y_oot, lgb_prob_shocked))
    
    print("\nDistribution Stability (Stress PSI):")
    print_psi_verdict("Logistic Regression", calculate_psi(lr_prob_train, lr_prob_shocked))
    print_psi_verdict("LightGBM Challenger", calculate_psi(lgb_prob_train, lgb_prob_shocked))
