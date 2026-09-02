# Credit Risk Model Validation & Stress Testing

A comprehensive credit risk analytics project that builds, validates, and stress-tests machine learning models for borrower default prediction using over **1 million historical loan records**.

The project compares a transparent **Logistic Regression scorecard** against a more complex **LightGBM model**, evaluates model stability using **Population Stability Index (PSI)**, and performs economic stress testing to assess performance under adverse scenarios.

---

## Project Objective

Financial institutions require credit risk models that are not only accurate but also:

* Interpretable
* Stable over time
* Robust under economic stress
* Suitable for regulatory environments

This project aims to answer:

> Can a simpler and more explainable model perform as well as a more complex machine learning model while remaining stable during adverse economic conditions?

---

## Dataset

* **Size:** 1,000,000+ historical loan records
* **Target Variable:** Borrower Default (0/1)
* **Example Features:**

  * Annual Income
  * Credit Score
  * Loan Amount
  * Debt-to-Income Ratio
  * Employment Information
  * Previous Credit History

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* Statistics

---

## Methodology

### 1. Data Preparation

* Missing value treatment
* Outlier handling
* Feature engineering
* Train/Test split
* Feature scaling (for Logistic Regression)

---

### 2. Model Development

#### Logistic Regression

* Highly interpretable
* Suitable for scorecard-style modeling
* Regulatory-friendly

#### LightGBM

* Gradient boosting framework
* Captures nonlinear relationships
* Higher model complexity

---

### 3. Model Validation

Models were evaluated using:

| Metric           | Purpose                |
| ---------------- | ---------------------- |
| AUC-ROC          | Discrimination ability |
| Gini Coefficient | Ranking performance    |
| KS Statistic     | Separation power       |
| PSI              | Population stability   |

---

## Economic Stress Testing

To simulate a recessionary environment:

* Borrower income reduced by **10%**
* Credit scores reduced by **25 points**

The stressed dataset was then passed through both models to assess:

* Stability
* Distributional drift
* Predictive robustness

---

## Population Stability Analysis (PSI)

Initial stress testing produced severe distributional drift:

| Stage                      | PSI  |
| -------------------------- | ---- |
| Before Feature Binning     | 1.67 |
| After 5-Bin Discretization | 0.25 |

Feature discretization improved model robustness by reducing sensitivity to extreme shifts in borrower characteristics.

---

## Results

| Model               | AUC   |
| ------------------- | ----- |
| Logistic Regression | 0.617 |
| LightGBM            | 0.617 |

### Key Finding

The simpler Logistic Regression model achieved performance comparable to LightGBM while offering:

* Better interpretability
* Easier regulatory explanation
* Lower operational complexity

---

## Project Workflow

```text
Raw Loan Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Logistic Regression & LightGBM
       ↓
Model Validation (AUC, KS, Gini)
       ↓
Economic Stress Testing
       ↓
PSI Analysis
       ↓
Feature Binning
       ↓
Final Model Selection
```

---

## Key Learnings

* Credit risk models must be evaluated beyond accuracy.
* Model stability is critical in changing economic conditions.
* Simpler models can often provide comparable performance with greater explainability.
* Stress testing and PSI analysis are essential components of model validation.

---

## Future Improvements

* Out-of-Time (OOT) Validation
* Probability of Default (PD) Calibration
* SHAP Explainability for LightGBM
* Scorecard Development
* IFRS-9 / Basel-style stress scenarios

---

## Author

**Anurag Parasar Mund**

Integrated MSc Mathematics
National Institute of Technology Rourkela

* LinkedIn: [Your LinkedIn]
* GitHub: [Your GitHub]

```

This README gives your project a **professional banking/model-risk feel**, which will look strong for **Piramal Finance, Sigmoid Analytics, analytics consulting firms, and credit-risk roles**. You can also add screenshots such as:
- ROC Curve
- PSI plots
- Feature importance charts
- Stress test comparison graphs

to make the GitHub repository even more impressive.
```

