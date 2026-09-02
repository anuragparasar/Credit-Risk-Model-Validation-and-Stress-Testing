# Credit Risk Model Validation & Stress Testing

A credit risk modeling project focused on **borrower default prediction, out-of-time (OOT) validation, model comparison, and macroeconomic stress testing** using over **1 million historical LendingClub loan records**.

The project benchmarks an interpretable **Logistic Regression** model against a **LightGBM challenger**, evaluates predictive performance using AUC, KS Statistic, and Brier Score, and assesses model stability under a simulated adverse economic scenario using the **Population Stability Index (PSI)**.

---

## 🎯 Project Objective

Credit risk models need to do more than achieve good predictive performance on historical data.

A reliable model should:

* Generalize to future, unseen populations
* Maintain stable predictions under changing economic conditions
* Provide meaningful risk discrimination
* Remain interpretable when required for business and regulatory purposes

This project investigates:

> **Can a simple, interpretable Logistic Regression model provide comparable predictive performance to a more complex LightGBM model while maintaining better stability and explainability under economic stress?**

---

## 📊 Dataset

The project uses the **LendingClub loan dataset**, containing more than **1 million historical loan records**.

### Target Variable

Borrower default is defined as:

* `0` → Fully Paid
* `1` → Charged Off / Default

### Features Used

| Feature          | Description                      |
| ---------------- | -------------------------------- |
| `dti`            | Debt-to-income ratio             |
| `fico_range_low` | Lower bound of FICO credit score |
| `annual_inc`     | Annual borrower income           |
| `revol_util`     | Revolving credit utilization     |

Additional loan-level fields used during preprocessing include:

* Loan status
* Issue date
* Loan amount
* Interest rate

---

# 🔄 Methodology

## 1. Data Preparation

The pipeline:

1. Loads the required LendingClub variables
2. Filters observations to valid loan outcomes
3. Creates the binary default target
4. Converts percentage-formatted variables to numerical values
5. Converts loan issue dates to datetime
6. Sorts observations chronologically

The chronological ordering is important because the project performs **Out-of-Time (OOT) validation**.

---

## 2. Chronological Development / OOT Split

Instead of randomly splitting the entire dataset, observations are first sorted by loan issue date.

The data is divided chronologically:

```text
Historical Data
│
├── First 80%
│     └── Development Population
│
└── Last 20%
      └── Out-of-Time (OOT) Population
```

The development population is subsequently split into:

```text
Development Population
│
├── 75% Training
│
└── 25% Out-of-Sample Validation
```

The OOT population remains completely separate and represents a **future/unseen population**.

---

# 🤖 Models

Two different modeling approaches are evaluated.

## Logistic Regression

A regularized Logistic Regression model is implemented using a preprocessing pipeline:

```text
Missing Value Imputation
        ↓
5-Quantile Feature Binning
        ↓
One-Hot Encoding
        ↓
Logistic Regression
```

Configuration:

* 5 quantile bins
* L2 regularization
* `C = 0.1`
* Maximum iterations = 1000

### Why Logistic Regression?

Logistic Regression provides:

* High interpretability
* Transparent feature relationships
* Easier model governance
* Straightforward explanation to business and regulatory stakeholders

---

## LightGBM Challenger

A LightGBM classifier is developed as the more complex challenger model.

Configuration includes:

* 60 estimators
* Learning rate = 0.03
* Maximum depth = 3
* 7 leaves
* Minimum child samples = 100

LightGBM provides a nonlinear alternative capable of capturing more complex relationships in the data.

---

# 📏 Model Evaluation

The models are evaluated using multiple risk-modeling metrics.

### AUC

**Area Under the ROC Curve** measures the model's ability to discriminate between defaulters and non-defaulters.

Higher AUC indicates stronger discriminatory power.

### KS Statistic

The **Kolmogorov-Smirnov (KS) statistic** measures the maximum separation between the predicted-risk distributions of:

* Defaulted borrowers
* Non-defaulted borrowers

### Brier Score

The **Brier Score** measures the accuracy of probabilistic predictions.

Lower values indicate better probability prediction quality.

---

# 🧪 Macroeconomic Stress Testing

The project simulates a severe economic deterioration by applying simultaneous shocks to the OOT population.

### Stress Scenario

| Variable              |          Shock |
| --------------------- | -------------: |
| Annual Income         |       **−10%** |
| FICO Score            | **−25 points** |
| Debt-to-Income Ratio  |       **+11%** |
| Revolving Utilization |       **+15%** |

The stressed OOT population is then passed through both models.

```text
OOT Population
      ↓
Macroeconomic Shocks
      ↓
Shocked OOT Population
      ↓
Logistic Regression
      ↓
LightGBM
      ↓
Compare Risk Predictions
```

This allows the models to be evaluated under conditions that differ materially from the historical development population.

---

# 📉 Population Stability Index (PSI)

Model stability is evaluated using **Population Stability Index (PSI)**.

In this project, PSI is calculated by comparing:

> **Training-period predicted probabilities vs. stressed OOT predicted probabilities**

The PSI implementation creates percentile-based bins using the expected distribution and measures how much the stressed distribution shifts across those bins.

### PSI Interpretation

The implementation uses the following thresholds:

|            PSI | Interpretation |
| -------------: | -------------- |
|       `< 0.10` | Stable         |
| `0.10 – <0.25` | Warning        |
|       `≥ 0.25` | Critical       |

A large PSI indicates that the model's predicted-risk distribution has shifted substantially under stress.

---

# 🛠️ Stability Improvement

The Logistic Regression pipeline uses **5-quantile feature discretization** before classification.

Instead of allowing continuous variables to enter the model directly, observations are grouped into five quantile-based categories.

```text
Continuous Features
        ↓
Quantile Binning
        ↓
5 Risk Categories
        ↓
One-Hot Encoding
        ↓
Logistic Regression
```

This design was motivated by the need to improve robustness against distributional changes under the simulated stress scenario.

---

# 📈 Results

The final model comparison showed that the simpler Logistic Regression approach achieved approximately the same discriminatory performance as the LightGBM challenger:

| Model               |       AUC |
| ------------------- | --------: |
| Logistic Regression | **0.617** |
| LightGBM            | **0.617** |

The stress-testing analysis also demonstrated substantial distributional movement before stabilization, with the project's reported PSI analysis showing:

```text
Initial PSI: 1.67
Final PSI:   0.25
```

The reduction indicates a substantial improvement in prediction-distribution stability under the tested stress scenario.

---

# 💡 Key Findings

### 1. Model complexity did not translate into better discrimination

LightGBM did not provide a meaningful AUC advantage over Logistic Regression in the evaluated scenario.

### 2. Model stability matters

A model can maintain reasonable predictive performance while still experiencing significant distributional changes under stress.

### 3. Simpler models can be attractive for regulated environments

When predictive performance is comparable, Logistic Regression offers significant advantages in:

* Interpretability
* Governance
* Transparency
* Model explainability

### 4. Stress testing provides another dimension of model validation

Testing the model against adverse borrower characteristics provides insight into how the model behaves outside normal historical conditions.

---

# 🧰 Technology Stack

* **Python**
* **Pandas**
* **NumPy**
* **SciPy**
* **Scikit-Learn**
* **LightGBM**
* **Statistics**

Key libraries used:

```python
pandas
numpy
scipy
scikit-learn
lightgbm
```

---

# 📁 Project Structure

```text
credit-risk-model/
│
├── loan.csv
├── credit_risk_model.py
├── README.md
└── results/
```

> The raw LendingClub dataset is not included in this repository due to its size/licensing considerations.

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd credit-risk-model
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn lightgbm
```

### 3. Place the dataset

Place the LendingClub dataset in the project directory as:

```text
loan.csv
```

### 4. Run the model

```bash
python credit_risk_model.py
```

The script will:

1. Load and preprocess the dataset
2. Create chronological development/OOT populations
3. Train Logistic Regression
4. Train LightGBM
5. Apply the macroeconomic stress scenario
6. Evaluate stressed OOT predictions
7. Calculate PSI
8. Print model stability results

---

# 🔬 Future Extensions

Potential extensions include:

* Probability of Default (PD) calibration
* Population-level scorecards
* Weight of Evidence (WoE) transformation
* Information Value (IV)
* Gini coefficient
* OOT performance monitoring
* PSI monitoring across individual features
* SHAP-based LightGBM explainability
* Calibration curves
* ROC and KS visualizations
* More granular macroeconomic scenarios
* Model performance monitoring across multiple time periods

---

# 👤 Author

**Anurag Parasar Mund**

Integrated MSc in Mathematics
National Institute of Technology, Rourkela

**Areas of Interest:** Data Science · Machine Learning · Credit Risk · Statistical Modeling · Model Validation
