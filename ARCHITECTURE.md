# System Architecture & Technical Specifications

## 1. System Overview
**EMIPredict AI** is an enterprise-grade financial risk assessment platform designed to automate loan underwriting, predict borrowers' maximum safe EMI, and deliver actionable counterfactual advice for subprime or borderline applicants.

---

## 2. Platform Use Case Architecture (Image 1)
```mermaid
flowchart LR
    subgraph EMIPredict_AI_Platform["EMIPredict AI Platform"]
        UC1(["Enter financial profile"])
        UC2(["Check EMI eligibility"])
        UC3(["Predict max safe EMI"])
        UC4(["Get approval-path advice"])
        UC5(["Explore data insights"])
        UC6(["Manage records (CRUD)"])
        UC7(["Monitor experiments"])
        UC8(["Promote model version"])
    end

    LO["Loan officer"] --> UC1
    LO --> UC2
    LO --> UC3
    LO --> UC4
    LO --> UC5

    AD["Admin"] --> UC5
    AD --> UC6

    ML["ML engineer"] --> UC7
    ML --> UC8
```

---

## 3. End-to-End Underwriting Sequence Diagram (Image 2)
```mermaid
sequenceDiagram
    autonumber
    actor Officer as Loan Officer
    participant App as Streamlit App
    participant ML as ML Service
    participant Adv as Advisor Engine

    Officer->>App: Enter Profile (22 Variables)
    App->>ML: Clean + Engineer Features (FOIR, DTI, Stability)
    ML->>ML: Predict Dual Targets (Eligibility & Max EMI)
    ML-->>App: Return Verdict + Max Safe EMI
    
    opt If Not Eligible or High_Risk
        App->>Adv: Trigger Counterfactual Search
        Adv->>ML: Re-score Variant Perturbations
        ML-->>Adv: Return Metrics & Attributions
        Adv-->>App: Deliver Smallest Fix + Risk Drivers
    end

    App-->>Officer: Render Decision Card + Approval-Path Advice
```

---

## 4. Machine Learning & MLOps Lifecycle Architecture (Image 3)
```mermaid
flowchart TD
    A["Load Data\n(400K rows across 5 scenarios)"] --> B["Clean Data\n(Nulls, Outliers, Bounds)"]
    B --> C["Engineer Features\n(FOIR, DTI, Disposable Cash)"]
    C --> D["Split + Scale\n(Train/Val/Test 70/15/15)"]
    D --> E["Train Models\n(3+ Clf & 3+ Reg)"]
    E --> F["Log Runs to MLflow\n(Hyperparameters, Metrics, ROC/RMSE)"]
    F --> G["Model Registry\n(Champion Version Promotion)"]
    G --> H["Streamlit Web App\n(Production Deployment)"]
    
    I["Advisor Engine\n(What-if Counterfactuals + Drivers)"] --> H
```

---

## 5. Domain Mathematical Formulations

### 5.1 Living Obligations & Disposable Income
$$\text{Total Expenses} = \text{Rent} + \text{School} + \text{College} + \text{Travel} + \text{Utilities} + \text{Misc}$$
$$\text{Disposable Income} = \text{Monthly Salary} - \text{Total Expenses} - \text{Current EMI}$$

### 5.2 Fixed Obligation to Income Ratio (FOIR)
$$\text{FOIR} = \frac{\text{Current EMI} + \text{Proposed Loan EMI} + \text{Rent}}{\text{Monthly Salary}}$$
*Prudent Banking Limit: $\le 50.0\%$ (Tiered up to $60.0\%$ for credit scores $> 750$).*

### 5.3 Algorithmic Safe Monthly EMI
$$\text{Max Safe EMI} = \min\left( \text{Monthly Salary} \times \text{FOIR}_{\text{cap}} - \text{Current EMI}, \; \text{Disposable Income} \times 0.75 \right)$$
Enforced between ₹500 and ₹50,000 INR.

### 5.4 Loan Amortization Formula
$$\text{EMI} = P \times \frac{r(1+r)^n}{(1+r)^n - 1}$$
Where $P$ is principal requested, $r$ is monthly scenario interest rate, and $n$ is repayment tenure in months.
