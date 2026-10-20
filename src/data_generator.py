"""
Synthetic Data Generation Engine for EMIPredict AI Platform.
Generates 400,000 realistic, domain-validated financial profiles across 5 lending scenarios
with exact target calculations following Indian banking credit underwriting norms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import os
from .config import (
    EMI_SCENARIOS,
    RAW_DATA_DIR,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET
)

def generate_emi_dataset(total_records_per_scenario: int = 80_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic 400,000 financial records (80k x 5 scenarios).
    Uses vectorized NumPy distributions for high generation speed and realistic correlation structures.
    """
    np.random.seed(seed)
    scenario_dfs = []

    print(f"Generating synthetic financial dataset ({total_records_per_scenario * len(EMI_SCENARIOS):,} records total)...")
    start_time = time.time()

    for scenario_name, specs in EMI_SCENARIOS.items():
        n = total_records_per_scenario
        
        # 1. Demographics
        age = np.random.randint(25, 61, size=n)
        gender = np.random.choice(["Male", "Female"], size=n, p=[0.58, 0.42])
        marital_status = np.where(age > 30, 
                                  np.random.choice(["Married", "Single"], size=n, p=[0.82, 0.18]),
                                  np.random.choice(["Single", "Married"], size=n, p=[0.60, 0.40]))
        education = np.random.choice(
            ["High School", "Graduate", "Post Graduate", "Professional"],
            size=n,
            p=[0.15, 0.50, 0.25, 0.10]
        )

        # 2. Employment & Income (correlated with education and age)
        edu_mult = {"High School": 0.7, "Graduate": 1.0, "Post Graduate": 1.35, "Professional": 1.65}
        base_salary_factor = np.array([edu_mult[e] for e in education])
        age_factor = 1.0 + (age - 25) * 0.015
        
        # Monthly gross salary (15k - 200k)
        raw_salary = np.random.lognormal(mean=10.3, sigma=0.45, size=n) * base_salary_factor * age_factor
        monthly_salary = np.clip(raw_salary, 15_000, 200_000).astype(int)

        employment_type = np.random.choice(
            ["Private", "Government", "Self-employed"],
            size=n,
            p=[0.65, 0.20, 0.15]
        )
        
        # Years of employment bounded by age - 21
        max_possible_exp = np.maximum(0, age - 21)
        years_of_employment = np.clip(
            (np.random.beta(2, 2, size=n) * max_possible_exp).astype(int),
            0, 35
        )

        company_type = np.random.choice(
            ["Startup", "Mid-size", "Large MNC", "Government"],
            size=n,
            p=[0.25, 0.35, 0.25, 0.15]
        )
        # Fix government employment consistency
        company_type = np.where(employment_type == "Government", "Government", company_type)

        # 3. Housing and Family
        house_type = np.random.choice(["Rented", "Own", "Family"], size=n, p=[0.45, 0.35, 0.20])
        rent_rate = np.random.uniform(0.12, 0.25, size=n)
        monthly_rent = np.where(house_type == "Rented", (monthly_salary * rent_rate).astype(int), 0)
        
        family_size = np.clip(np.random.poisson(lam=3.2, size=n), 1, 7)
        dependents = np.clip(np.random.binomial(n=family_size - 1, p=0.6), 0, 5)

        # 4. Monthly Financial Obligations
        has_kids = (dependents > 0)
        school_fees = np.where(has_kids, np.random.uniform(1_500, 12_000, size=n) * dependents, 0).astype(int)
        college_fees = np.where((dependents > 0) & (age > 42), np.random.uniform(3_000, 25_000, size=n), 0).astype(int)
        travel_expenses = np.clip(monthly_salary * np.random.uniform(0.03, 0.08, size=n), 1_000, 15_000).astype(int)
        groceries_utilities = np.clip(monthly_salary * np.random.uniform(0.10, 0.22, size=n) + (family_size * 1_200), 4_000, 35_000).astype(int)
        other_monthly_expenses = np.clip(monthly_salary * np.random.uniform(0.02, 0.08, size=n), 1_000, 20_000).astype(int)

        # 5. Financial Status & Credit History
        existing_loans = np.random.choice([0, 1, 2, 3], size=n, p=[0.40, 0.35, 0.18, 0.07])
        emi_rate = np.random.uniform(0.10, 0.25, size=n)
        current_emi_amount = np.where(
            existing_loans > 0,
            np.clip(monthly_salary * emi_rate * existing_loans * 0.5, 2_000, 40_000).astype(int),
            0
        )
        
        # Credit score: skewed normal between 300 and 850
        credit_score_raw = np.random.normal(loc=680, scale=85, size=n)
        # Shift down slightly for high existing loans
        credit_score_raw -= existing_loans * 25
        credit_score = np.clip(credit_score_raw, 300, 850).astype(int)

        # Bank balance & emergency fund
        savings_prop = np.random.uniform(0.5, 3.5, size=n)
        bank_balance = np.clip(monthly_salary * savings_prop, 5_000, 600_000).astype(int)
        emergency_fund = np.clip(bank_balance * np.random.uniform(0.1, 0.8, size=n), 0, 400_000).astype(int)

        # 6. Loan Application Details
        requested_amount = np.random.randint(
            specs["min_amount"], specs["max_amount"] + 1, size=n
        )
        requested_tenure = np.random.choice(
            range(specs["min_tenure"], specs["max_tenure"] + 1, 3 if specs["max_tenure"] > 12 else 1),
            size=n
        )

        # 7. Ground Truth Calculation (FinTech Underwriting Logic)
        total_monthly_expenses = (
            monthly_rent + school_fees + college_fees + 
            travel_expenses + groceries_utilities + other_monthly_expenses
        )
        
        disposable_income = monthly_salary - total_monthly_expenses - current_emi_amount
        
        # Banking FOIR Cap: Standard 40% - 60% based on credit score tier
        cs_factor = (credit_score - 300) / 550.0 # 0.0 to 1.0
        max_foir = 0.35 + (0.25 * cs_factor) # 0.35 to 0.60
        max_total_emi = monthly_salary * max_foir
        available_emi_cap = np.maximum(0, max_total_emi - current_emi_amount)
        
        # Safe EMI cannot exceed 75% of actual unallocated disposable income
        cashflow_cap = np.maximum(0, disposable_income * 0.75)
        
        raw_max_emi = np.minimum(available_emi_cap, cashflow_cap)
        # Enforce target bounds 500 to 50,000 INR
        max_monthly_emi = np.clip(raw_max_emi, 500, 50_000).astype(int)

        # Estimate monthly installment for the requested loan
        avg_interest = np.mean(specs["interest_rate_range"])
        # Standard EMI amortization formula: P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_r = avg_interest / 12.0
        tenure_n = requested_tenure
        factor = (1 + monthly_r) ** tenure_n
        estimated_req_emi = requested_amount * (monthly_r * factor) / (factor - 1)

        # Target 1: EMI Eligibility classification
        # Three classes: Eligible, High_Risk, Not_Eligible
        # Strict underwriting criteria
        not_eligible_mask = (
            (credit_score < 560) |
            (disposable_income < 0.08 * monthly_salary) |
            (estimated_req_emi > max_monthly_emi * 1.30) |
            ((current_emi_amount + estimated_req_emi) > monthly_salary * 0.65) |
            (disposable_income < estimated_req_emi * 0.85)
        )
        
        high_risk_mask = (
            ~not_eligible_mask & (
                (credit_score < 660) |
                (estimated_req_emi > max_monthly_emi * 0.95) |
                ((current_emi_amount + estimated_req_emi) > monthly_salary * 0.48) |
                (emergency_fund < total_monthly_expenses * 0.8) |
                (years_of_employment < 1)
            )
        )
        
        emi_eligibility = np.where(
            not_eligible_mask,
            "Not_Eligible",
            np.where(high_risk_mask, "High_Risk", "Eligible")
        )

        df_scenario = pd.DataFrame({
            # Demographics
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            # Employment
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            # Housing & Family
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            # Monthly obligations
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            # Financial Status
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            # Loan details
            "emi_scenario": scenario_name,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            # Targets
            CLASSIFICATION_TARGET: emi_eligibility,
            REGRESSION_TARGET: max_monthly_emi
        })
        scenario_dfs.append(df_scenario)
        print(f"Generated {n:,} records for scenario: {scenario_name}")

    full_df = pd.concat(scenario_dfs, ignore_index=True)
    # Shuffle
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    elapsed = time.time() - start_time
    print(f"Complete dataset generated: {len(full_df):,} records in {elapsed:.2f} seconds.")
    return full_df


def save_dataset(df: pd.DataFrame, output_dir: Path = RAW_DATA_DIR):
    """
    Saves dataset to CSV (and Parquet if engine available).
    Also creates a sample for immediate UI responsiveness.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "emi_dataset_400k.csv"
    sample_path = output_dir / "emi_dataset_sample_50k.csv"

    print("Saving 400K dataset to CSV...")
    df.to_csv(csv_path, index=False)
    print(f"Saved complete 400k dataset: {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)")

    df.sample(n=min(50_000, len(df)), random_state=42).to_csv(sample_path, index=False)
    print(f"Saved 50k representative sample: {sample_path} ({sample_path.stat().st_size / (1024*1024):.2f} MB)")

    parquet_path = output_dir / "emi_dataset_400k.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet format: {parquet_path}")
    except Exception as e:
        print(f"Note: Parquet engine not available, using standard CSV: {e}")

    return csv_path, sample_path

if __name__ == "__main__":
    df = generate_emi_dataset(total_records_per_scenario=80_000)
    save_dataset(df)
