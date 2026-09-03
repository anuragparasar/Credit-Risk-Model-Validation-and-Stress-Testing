**Credit Risk Model Validation & Macroeconomic Stress Testing**

A comprehensive credit risk modeling framework predicting borrower defaults using historical LendingClub loan records. This project benchmarks an interpretable Logistic Regression model against a LightGBM challenger, evaluating predictive performance (AUC, KS Statistic, Brier Score) and assessing model stability under multi-tier economic downturns using the Population Stability Index (PSI).

## 🎯 Project Overview & Methodology

Credit models must generalize to unseen populations and remain stable during economic shifts. To test this, the project utilizes rigorous chronological out-of-time (OOT) validation.

*   **Dataset Setup**: Over 1.3 million records are split chronologically into an 80% In-Time development set and a 20% Out-of-Time future population.
*   **Feature Engineering**: The pipeline leverages core borrower risk indicators: Debt-to-Income, lowest FICO score, annual income, and revolving utilization.
*   **Stability Enhancements**: Continuous features undergo 5-quantile binning prior to Logistic Regression training to explicitly improve robustness against distributional changes.

## 🤖 Modeling & Macroeconomic Stress Testing

Two distinct models—a transparent Logistic Regression and a non-linear LightGBM classifier—are subjected to multi-tier synthetic economic shocks applied to the OOT dataset. 

*   **Mild Stress**: Simulates a baseline slowdown with a 5% income drop and 10-point FICO reduction.
*   **Moderate Stress**: Mirrors a typical recession featuring a 10% income drop and 25-point FICO reduction.
*   **Severe Stress (CCAR)**: Replicates a deep crisis via a 20% income drop, 50-point FICO drop, and 30% higher revolving utilization.
*   **Target Adjustment**: A binomial distribution artificially injects stress defaults, inflating the OOT default rate from 21.79% to 29.62%.

## 📈 Performance Results & PSI Drift

Model stability is tracked using PSI (Stable: <0.10, Warning: 0.10–0.25, Critical: >0.25). 

*   **Baseline Performance**: Both algorithms demonstrated comparable unshocked discrimination (AUC ~0.62) and excellent stability (PSI ~0.01).
*   **Stress Degradation**: Under severe stress, both models experienced critical distribution drift; Logistic Regression reached a PSI of 0.87, while LightGBM hit 0.78.
*   **Shocked Accuracy**: When evaluated against the inflated stress default rate, AUC plummeted to ~0.58 for both models, highlighting the impact of compounding economic shocks.

## 💡 Key Findings & Tech Stack

*   **Complexity vs. Value**: The complex LightGBM algorithm did not provide a meaningful discriminatory advantage over the simpler, highly interpretable Logistic Regression model.
*   **Stress Vulnerability**: Even models with robust feature discretization suffer significant predictive degradation under CCAR-level severity.

**Technology Stack:** Python, Pandas, NumPy, SciPy, Scikit-Learn, and LightGBM.  
**Author:** Developed by Anurag Parasar Mund, National Institute of Technology, Rourkela.
