"""
Feature Engineering Pipeline for EMIPredict AI.
Calculates high-impact domain financial ratios, credit risk tiers,
employment stability indices, and interaction variables.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Standard interest rate estimates per scenario for amortization
SCENARIO_INTEREST_RATES = {
    "E-commerce Shopping EMI": 0.16,
    "Home Appliances EMI": 0.14,
    "Vehicle EMI": 0.10,
    "Personal Loan EMI": 0.15,
    "Education EMI": 0.10
}

def calculate_estimated_emi(amount: np.ndarray, tenure_months: np.ndarray, scenario_series: pd.Series) -> np.ndarray:
    """
    Computes standard monthly amortized EMI: P * r * (1+r)^n / ((1+r)^n - 1)
    """
    rates = scenario_series.map(SCENARIO_INTEREST_RATES).fillna(0.12).values
    monthly_r = rates / 12.0
    n = np.maximum(1, tenure_months)
    factor = (1.0 + monthly_r) ** n
    # For small n or zero r
    emi = np.where(monthly_r > 0, amount * (monthly_r * factor) / (factor - 1.0), amount / n)
    return np.round(emi, 2)


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw applicant attributes into domain-rich risk metrics.
    Works seamlessly on both pandas DataFrame batches and single-row records.
    """
    data = df.copy()

    # 1. Total Living Obligations
    data["total_monthly_expenses"] = (
        data["monthly_rent"] +
        data["school_fees"] +
        data["college_fees"] +
        data["travel_expenses"] +
        data["groceries_utilities"] +
        data["other_monthly_expenses"]
    )

    # 2. Net Unencumbered Cashflow
    data["disposable_income"] = (
        data["monthly_salary"] - data["total_monthly_expenses"] - data["current_emi_amount"]
    )

    # 3. Estimated Proposed Loan Installment
    data["estimated_req_emi"] = calculate_estimated_emi(
        data["requested_amount"].values,
        data["requested_tenure"].values,
        data["emi_scenario"]
    )

    # 4. Standard Banking Risk Ratios
    salary_safe = np.maximum(1.0, data["monthly_salary"].values)
    expenses_safe = np.maximum(1.0, data["total_monthly_expenses"].values)
    req_amount_safe = np.maximum(1.0, data["requested_amount"].values)

    # Debt-To-Income (existing obligations)
    data["debt_to_income_ratio"] = np.round(data["current_emi_amount"] / salary_safe, 4)

    # Expense-To-Income
    data["expense_to_income_ratio"] = np.round(data["total_monthly_expenses"] / salary_safe, 4)

    # Fixed Obligation to Income Ratio (FOIR) including proposed new loan
    data["fixed_obligation_ratio"] = np.round(
        (data["current_emi_amount"] + data["estimated_req_emi"] + data["monthly_rent"]) / salary_safe, 4
    )

    # Emergency Fund Coverage in Months of living expenses
    data["emergency_fund_months"] = np.round(data["emergency_fund"] / expenses_safe, 2)

    # Liquid Assets to Requested Loan Ratio
    total_liquid_reserves = data["bank_balance"] + data["emergency_fund"]
    data["savings_to_loan_ratio"] = np.round(total_liquid_reserves / req_amount_safe, 4)

    # 5. Composite Employment Stability Score (0.0 to 1.0)
    company_weight = data["company_type"].map({
        "Government": 1.0,
        "Large MNC": 0.85,
        "Mid-size": 0.70,
        "Startup": 0.55
    }).fillna(0.60)
    
    emp_type_weight = data["employment_type"].map({
        "Government": 1.0,
        "Private": 0.80,
        "Self-employed": 0.65
    }).fillna(0.75)

    exp_normalized = np.clip(data["years_of_employment"] / 15.0, 0.0, 1.0)
    data["employment_stability_score"] = np.round(
        (company_weight * 0.40) + (emp_type_weight * 0.35) + (exp_normalized * 0.25), 4
    )

    # 6. Credit Risk Tier (Categorical/Ordinal)
    cs = data["credit_score"]
    data["credit_risk_tier"] = np.select(
        [cs >= 750, (cs >= 680) & (cs < 750), (cs >= 600) & (cs < 680), cs < 600],
        ["Tier_1_Excellent", "Tier_2_Good", "Tier_3_Fair", "Tier_4_Poor"],
        default="Tier_3_Fair"
    )

    # 7. Interaction & Capacity Features
    data["loan_to_annual_income"] = np.round(data["requested_amount"] / (salary_safe * 12.0), 4)
    data["credit_dti_interaction"] = np.round(data["credit_score"] * (1.0 - np.clip(data["debt_to_income_ratio"], 0, 1.0)), 2)

    return data
