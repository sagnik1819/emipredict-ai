"""
Data Cleaning, Quality Assessment, and Preprocessing Pipeline for EMIPredict AI.
Handles missing data imputation, anomaly detection, categorical encoding,
feature scaling, and leak-free Train/Val/Test stratification.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ENGINEERED_FEATURES,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    PROCESSED_DATA_DIR,
    MODELS_DIR
)
from .feature_engineering import engineer_financial_features

class DataQualityAssessor:
    """Performs rigorous automated banking data quality assessment."""

    @staticmethod
    def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "missing_values_total": int(df.isnull().sum().sum()),
            "missing_per_column": df.isnull().sum().to_dict(),
            "duplicate_records": int(df.duplicated().sum()),
            "anomalies": {
                "negative_salaries": int((df["monthly_salary"] <= 0).sum()) if "monthly_salary" in df else 0,
                "invalid_credit_scores": int(((df["credit_score"] < 300) | (df["credit_score"] > 850)).sum()) if "credit_score" in df else 0,
                "negative_rent": int((df["monthly_rent"] < 0).sum()) if "monthly_rent" in df else 0,
            },
            "class_distribution": df[CLASSIFICATION_TARGET].value_counts().to_dict() if CLASSIFICATION_TARGET in df else {},
            "scenario_distribution": df["emi_scenario"].value_counts().to_dict() if "emi_scenario" in df else {}
        }
        return report


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans inconsistencies, removes duplicates, and ensures valid numerical bounds.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    # Null handling (imputation strategy)
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in cat_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    # Outlier / logical clipping
    if "credit_score" in cleaned:
        cleaned["credit_score"] = np.clip(cleaned["credit_score"], 300, 850)
    if "monthly_salary" in cleaned:
        cleaned["monthly_salary"] = np.clip(cleaned["monthly_salary"], 10_000, 500_000)

    return cleaned


class PreprocessingPipeline:
    """
    Leak-free feature transformer for training and inference.
    Fits ColumnTransformer on training split and transforms val/test/production instances.
    """
    def __init__(self):
        self.column_transformer = None
        self.class_mapping = {"Eligible": 0, "High_Risk": 1, "Not_Eligible": 2}
        self.inv_class_mapping = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
        self.feature_names_out = []
        self.fitted = False

    def build_transformer(self, num_cols, cat_cols):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ],
            remainder="drop"
        )

    def fit(self, X: pd.DataFrame):
        # Auto-engineer if not already present
        if "disposable_income" not in X.columns:
            X = engineer_financial_features(X)

        cat_cols = [c for c in CATEGORICAL_FEATURES + ["credit_risk_tier"] if c in X.columns]
        num_cols = [c for c in NUMERICAL_FEATURES + [
            "total_monthly_expenses", "disposable_income", "debt_to_income_ratio",
            "expense_to_income_ratio", "fixed_obligation_ratio", "emergency_fund_months",
            "savings_to_loan_ratio", "employment_stability_score", "loan_to_annual_income",
            "credit_dti_interaction", "estimated_req_emi"
        ] if c in X.columns]

        self.build_transformer(num_cols, cat_cols)
        self.column_transformer.fit(X)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.fitted = True

        # Extract transformed feature names
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        encoded_cats = list(cat_encoder.get_feature_names_out(cat_cols))
        self.feature_names_out = num_cols + encoded_cats
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transforming.")
        if "disposable_income" not in X.columns:
            X = engineer_financial_features(X)
        return self.column_transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: Path = MODELS_DIR / "preprocessor.joblib"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor successfully saved to: {filepath}")

    @staticmethod
    def load(filepath: Path = MODELS_DIR / "preprocessor.joblib") -> "PreprocessingPipeline":
        return joblib.load(filepath)


def prepare_data_splits(
    df: pd.DataFrame, 
    test_size: float = 0.15, 
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates 70/15/15 stratified train-val-test split based on eligibility and scenario.
    """
    df_clean = clean_raw_data(df)
    df_feat = engineer_financial_features(df_clean)

    # Composite stratify key
    stratify_col = df_feat[CLASSIFICATION_TARGET].astype(str) + "_" + df_feat["emi_scenario"].astype(str)

    # First split off test set (15%)
    train_val, test_df = train_test_split(
        df_feat, test_size=test_size, random_state=seed, stratify=stratify_col
    )

    # Then split val set from train_val (val_size / (1 - test_size) = ~0.176 for 15% overall)
    stratify_val = train_val[CLASSIFICATION_TARGET].astype(str) + "_" + train_val["emi_scenario"].astype(str)
    adj_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val, test_size=adj_val_size, random_state=seed, stratify=stratify_val
    )

    print(f"Stratified Splits: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
