# EMIPredict AI Configuration & Domain Schemas
"""
Central configuration module for EMIPredict AI.
Defines dataset schemas, financial domain thresholds, model hyperparameter grids,
and system paths.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "applicant_records.db"
MLRUNS_DIR = BASE_DIR / "mlruns"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 5 EMI Scenarios with loan bounds and typical tenures
EMI_SCENARIOS = {
    "E-commerce Shopping EMI": {
        "min_amount": 10_000,
        "max_amount": 200_000,
        "min_tenure": 3,
        "max_tenure": 24,
        "target_records": 80_000,
        "interest_rate_range": (0.12, 0.20)
    },
    "Home Appliances EMI": {
        "min_amount": 20_000,
        "max_amount": 300_000,
        "min_tenure": 6,
        "max_tenure": 36,
        "target_records": 80_000,
        "interest_rate_range": (0.10, 0.18)
    },
    "Vehicle EMI": {
        "min_amount": 80_000,
        "max_amount": 1_500_000,
        "min_tenure": 12,
        "max_tenure": 84,
        "target_records": 80_000,
        "interest_rate_range": (0.08, 0.14)
    },
    "Personal Loan EMI": {
        "min_amount": 50_000,
        "max_amount": 1_000_000,
        "min_tenure": 12,
        "max_tenure": 60,
        "target_records": 80_000,
        "interest_rate_range": (0.11, 0.22)
    },
    "Education EMI": {
        "min_amount": 50_000,
        "max_amount": 500_000,
        "min_tenure": 6,
        "max_tenure": 48,
        "target_records": 80_000,
        "interest_rate_range": (0.08, 0.13)
    }
}

# 22 Input Variables specification
CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "emi_scenario"
]

NUMERICAL_FEATURES = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure"
]

ENGINEERED_FEATURES = [
    "total_monthly_expenses",
    "disposable_income",
    "debt_to_income_ratio",        # DTI
    "expense_to_income_ratio",     # ETI
    "fixed_obligation_ratio",      # FOIR (existing emi + rent + requested emi est / salary)
    "emergency_fund_months",       # emergency fund / total monthly expenses
    "savings_to_loan_ratio",       # (bank_balance + emergency_fund) / requested_amount
    "employment_stability_score",  # composite score based on employment type & years
    "credit_risk_tier",            # ordinal tier from credit score
    "estimated_proposed_emi"       # approximate EMI for requested loan at standard scenario interest
]

# Targets
CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

# Eligibility classes
ELIGIBILITY_CLASSES = ["Eligible", "High_Risk", "Not_Eligible"]

# Metric thresholds required by specification
TARGET_ACCURACY = 0.90
TARGET_REGRESSION_RMSE = 2000.0
