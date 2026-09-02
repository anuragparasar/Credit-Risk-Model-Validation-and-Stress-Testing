import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from scipy.stats import ks_2samp
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SYNTHETIC DATA GENERATION & SPLITTING
# ==========================================
def generate_credit_data(n_samples=10000):
    """Generates synthetic loan data mimicking retail credit portfolios."""
    np.random.seed(42)
    
    # Simulating features: DTI (Debt-to-Income), FICO, Revolving Utilization, Income
    dti = np.random.normal(loc=0.3, scale=0.1, size=n_samples)
    fico = np.random.normal(loc=650, scale=50, size=n_samples)
    utilization = np.random.uniform(low=0.1, high=0.9, size=n_samples)
    income = np.random.lognormal(mean=11, sigma=0.5, size=n_samples)
    
    # Introducing temporal drift for the Out-of-Time (OOT) set
    drift_mask = np.arange(n_samples) > int(n_samples * 0.8)
    fico[drift_mask] -= 20 # Simulating an economic downturn in later months
    
    # Generating default probabilities via logistic function
    logit = -0.01 * fico + 3.0 * dti + 1.0 * utilization - 0.00001 * income + 5.0
    prob_default = 1 / (1 + np.exp(-logit))
    
    # Binarize default based on probability
    default = np.random.binomial(1, prob_default)
    
    df = pd.DataFrame({
        'dti': dti, 'fico': fico, 'utilization': utilization, 
        'income': income, 'default': default
    })
    
    # Temporal Split: 80% In-Time (Train + Out-of-Sample), 20% Out-of-Time (OOT)
    df_in_time = df.iloc[:int(n_samples * 0.8)]
    df_oot = df.iloc[int(n_samples * 0.8):]
    
    return df_in_time, df_oot

# ==========================================
# 2. VALIDATION METRICS & STABILITY (PSI/KS)
# ==========================================
def calculate_psi(expected, actual, bins=10):
    """
    Calculates the Population Stability Index (PSI) to measure model drift[cite: 1].
    Mathematically measures the divergence between two probability distributions.
    """
    expected_pct = np.histogram(expected, bins=bins, range=(0, 1))[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins, range=(0, 1))[0] / len(actual)
    
    # Replace zeros to avoid log(0) and division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)

def calculate_ks_statistic(y_true, y_prob):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic[cite: 1].
    Measures the maximum separation between the empirical CDFs of goods and bads.
    """
    prob_goods = y_prob[y_true == 0]
    prob_bads = y_prob[y_true == 1]
    ks_stat, _ = ks_2samp(prob_bads, prob_goods)
    return ks_stat

def evaluate_model(y_true, y_prob, threshold=0.5):
    """Evaluates standard credit risk metrics: AUC, Gini, KS, Precision, Recall[cite: 1]."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    ks = calculate_ks_statistic(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return {'AUC': auc, 'Gini': gini, 'KS_Stat': ks, 'Precision': precision, 'Recall': recall}

# ==========================================
# 3. MODEL DEVELOPMENT & CHALLENGER AUDIT
# ==========================================
if __name__ == "__main__":
    print("--- 1. Data Extraction & Sampling Methodology ---")
    df_in_time, df_oot = generate_credit_data()
    
    # Out-of-Sample (OOS) validation split[cite: 1]
    X_train, X_oos, y_train, y_oos = train_test_split(
        df_in_time.drop('default', axis=1), df_in_time['default'], 
        test_size=0.25, random_state=42
    )
    X_oot, y_oot = df_oot.drop('default', axis=1), df_oot['default']
    
    print("Train Size:", len(X_train), "| OOS Size:", len(X_oos), "| OOT Size:", len(X_oot))
    
    print("\n--- 2. Baseline Model: Logistic Regression Scorecard ---")
    # For a true scorecard, variables should be binned via Weight of Evidence (WoE)
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train, y_train)
    
    lr_prob_train = baseline_model.predict_proba(X_train)[:, 1]
    lr_prob_oos = baseline_model.predict_proba(X_oos)[:, 1]
    lr_prob_oot = baseline_model.predict_proba(X_oot)[:, 1]
    
    print("Baseline OOS Metrics:", evaluate_model(y_oos, lr_prob_oos))
    print("Baseline OOT Metrics:", evaluate_model(y_oot, lr_prob_oot))
    
    print("\n--- 3. Challenger Model: LightGBM ---")
    # Benchmarking model outputs against a challenger model[cite: 1]
    challenger_model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, is_unbalance=True)
    challenger_model.fit(X_train, y_train)
    
    lgb_prob_oos = challenger_model.predict_proba(X_oos)[:, 1]
    lgb_prob_oot = challenger_model.predict_proba(X_oot)[:, 1]
    
    print("Challenger OOS Metrics:", evaluate_model(y_oos, lgb_prob_oos))
    print("Challenger OOT Metrics:", evaluate_model(y_oot, lgb_prob_oot))
    
    print("\n--- 4. Model Drift & Stability Assessment (PSI) ---")
    # Identifying model risks including model drift via PSI[cite: 1]
    psi_baseline = calculate_psi(lr_prob_train, lr_prob_oot)
    psi_challenger = calculate_psi(challenger_model.predict_proba(X_train)[:, 1], lgb_prob_oot)
    
    print(f"Baseline Score PSI (Train vs OOT): {psi_baseline:.4f}")
    print(f"Challenger Score PSI (Train vs OOT): {psi_challenger:.4f}")
    
    if psi_baseline >= 0.1 or psi_challenger >= 0.1:
        print("\nWARNING [Validation Finding]: Moderate to severe model drift detected in OOT sample.")
        print("Action Item: Review macroeconomic shifts affecting FICO distributions in recent origination cohorts.")