# EMIPredict AI - Comprehensive Technical & Business Capstone Report

**Domain:** FinTech & Retail Banking Underwriting  
**Scale:** 400,000 Records across 5 Lending Categories  
**Authors:** Senior Machine Learning & Financial Risk Engineering Team  

---

## 1. Executive Summary
In retail banking and consumer lending, default risks frequently stem from inadequate holistic cash flow assessment and rigid approval rubrics. Traditional credit decisioning often relies solely on bureau scores while overlooking granular monthly obligations (e.g., school fees, rentals, utility burdens).

**EMIPredict AI** solves this critical challenge by deploying a production-ready dual-engine machine learning platform that:
1. Performs real-time **EMI Eligibility Classification** (`Eligible`, `High_Risk`, `Not_Eligible`).
2. Computes the borrower's exact **Maximum Safe Monthly EMI Capacity** (`max_monthly_emi`) using debt-capacity and cash-flow regression.
3. Automatically triggers an **Approval Advisor** for declined or high-risk borrowers to calculate the *minimal viable counterfactual fix* (e.g. tenure elongation, downpayment adjustment, debt paydown).
4. Provides continuous experiment tracking via **MLflow** and full database **CRUD lifecycle operations**.

---

## 2. Dataset Architecture & Quality Audit
The dataset consists of **400,000 records** distributed uniformly across 5 realistic consumer lending verticals (80,000 profiles each):

| Scenario | Loan Range | Tenure Range | Primary Collateral / Purpose | Target Vol |
| :--- | :--- | :--- | :--- | :--- |
| **E-commerce Shopping EMI** | ₹10,000 - ₹200,000 | 3 - 24 Months | Unsecured consumer electronics | 80,000 |
| **Home Appliances EMI** | ₹20,000 - ₹300,000 | 6 - 36 Months | Consumer durables / appliances | 80,000 |
| **Vehicle EMI** | ₹80,000 - ₹1,500,000 | 12 - 84 Months | Hypothecated 2-wheeler / 4-wheeler | 80,000 |
| **Personal Loan EMI** | ₹50,000 - ₹1,000,000 | 12 - 60 Months | General liquidity & emergency cash | 80,000 |
| **Education EMI** | ₹50,000 - ₹500,000 | 6 - 48 Months | Higher degree / vocational tuition | 80,000 |

### 22-Variable Input Schema
* **Personal Demographics (4):** `age` (25–60), `gender`, `marital_status`, `education` (High School, Graduate, Post Graduate, Professional).
* **Employment & Income (4):** `monthly_salary` (₹15,000–₹200,000), `employment_type` (Private, Government, Self-employed), `years_of_employment` (0–35), `company_type` (Startup, Mid-size, Large MNC, Government).
* **Housing & Household (4):** `house_type` (Rented, Own, Family), `monthly_rent`, `family_size`, `dependents`.
* **Monthly Obligations (5):** `school_fees`, `college_fees`, `travel_expenses`, `groceries_utilities`, `other_monthly_expenses`.
* **Credit & Liquidity Status (5):** `existing_loans`, `current_emi_amount`, `credit_score` (300–850), `bank_balance`, `emergency_fund`.
* **Loan Request Details (3):** `emi_scenario`, `requested_amount`, `requested_tenure`.

---

## 3. Feature Engineering & Domain Ratios
Standard raw inputs are transformed into high-signal banking risk indicators:
1. **Total Monthly Expenses:** Aggregates rent, schooling, living, travel, and utility expenses.
2. **Net Disposable Income:** Gross salary minus all living expenses and active EMI commitments.
3. **Debt-to-Income Ratio (DTI):** Active EMI burden relative to monthly income.
4. **Expense-to-Income Ratio (ETI):** Essential living cost consumption rate.
5. **Fixed Obligation to Income Ratio (FOIR):** Combined existing EMI, rent, and proposed new installment divided by salary.
6. **Emergency Reserve Coverage:** Liquid savings expressed in months of average living expenses.
7. **Employment Stability Score:** Weighted index combining employer stature (Government > MNC > Mid-size > Startup) and tenure.

---

## 4. Machine Learning Benchmark Results

Both ML problems were evaluated across 3 distinct candidate algorithms on an independent stratified test split (6,000 test records):

### 4.1 Classification: EMI Eligibility Prediction (Target: Accuracy > 90%)
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (OVR) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 89.20% | 0.8143 | 0.8145 | 0.8143 | 0.9707 | Baseline |
| **Random Forest Classifier** | 97.23% | 0.9548 | 0.9552 | 0.9550 | 0.9981 | Passed |
| **Gradient Boosting Classifier (Champion)** | **99.10%** | **0.9865** | **0.9868** | **0.9866** | **0.9998** | **Promoted to Production** |

*Target Evaluation:* Exceeds benchmark hurdle (>90%) by **+9.10%**.

### 4.2 Regression: Maximum Safe Monthly EMI (Target: RMSE < ₹2,000 INR)
| Model | RMSE (INR) | MAE (INR) | $R^2$ Score | MAPE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Linear Regression** | ₹2,469.4 | ₹1,851.6 | 0.9502 | 14.2% | Baseline |
| **Random Forest Regressor** | ₹491.9 | ₹118.0 | 0.9980 | 1.8% | Passed |
| **Gradient Boosting Regressor (Champion)** | **₹341.7** | **₹106.5** | **0.9990** | **1.2%** | **Promoted to Production** |

*Target Evaluation:* Crushes benchmark hurdle (< ₹2,000) by delivering an ultra-precise RMSE of **₹341.7**.

---

## 5. MLOps Lifecycle & MLflow Registry
* **Tracking Server:** Automatic logging of model artifacts, parameter sets (learning rate, tree depths, regularization), and multi-metric evaluations.
* **Model Registry:** Staging and versioning console enabling ML engineers to inspect challenger versions and promote champion models with zero downtime.

---

## 6. Business Impact & ROI Analysis
1. **80% Turnaround Time Reduction:** Instantaneous dual-prediction reduces manual underwriter review from 25–45 minutes down to < 50 milliseconds.
2. **Reduced Default Rates:** Cash-flow-backed FOIR limits prevent over-leveraging of marginal borrowers.
3. **Customer Retention via Advisor:** Instead of outright rejections, loan officers can offer instant restructured terms (e.g. extending tenure from 24 to 36 months), converting declines into performing loan assets.
