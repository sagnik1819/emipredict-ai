"""
Comprehensive Unit Test Suite for EMIPredict AI Platform.
Tests feature engineering, preprocessing, model inference,
CRUD database operations, and the Approval Advisor counterfactual engine.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.config import MODELS_DIR, EMI_SCENARIOS
from src.feature_engineering import engineer_financial_features
from src.preprocessing import clean_raw_data, PreprocessingPipeline
from src.advisor_engine import ApprovalAdvisor
from src.crud_service import ApplicantService, init_db

class TestEMIPredictPipeline(unittest.TestCase):

    def setUp(self):
        init_db()
        self.sample_row = {
            "age": 32, "gender": "Male", "marital_status": "Married", "education": "Graduate",
            "monthly_salary": 60000, "employment_type": "Private", "years_of_employment": 5,
            "company_type": "Large MNC", "house_type": "Rented", "monthly_rent": 12000,
            "family_size": 3, "dependents": 1, "school_fees": 3000, "college_fees": 0,
            "travel_expenses": 3000, "groceries_utilities": 10000, "other_monthly_expenses": 2000,
            "existing_loans": 1, "current_emi_amount": 4000, "credit_score": 720,
            "bank_balance": 100000, "emergency_fund": 50000, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 250000, "requested_tenure": 24
        }
        self.df_sample = pd.DataFrame([self.sample_row])

    def test_feature_engineering(self):
        feat_df = engineer_financial_features(self.df_sample)
        self.assertIn("disposable_income", feat_df.columns)
        self.assertIn("debt_to_income_ratio", feat_df.columns)
        self.assertIn("fixed_obligation_ratio", feat_df.columns)
        self.assertIn("estimated_req_emi", feat_df.columns)
        
        # Verify disposable income math: 60000 - (12000+3000+0+3000+10000+2000) - 4000 = 26000
        expected_disposable = 60000 - 30000 - 4000
        self.assertEqual(feat_df["disposable_income"].iloc[0], expected_disposable)

    def test_preprocessor_and_models(self):
        prep_path = MODELS_DIR / "preprocessor.joblib"
        clf_path = MODELS_DIR / "classification" / "best_classification_model.joblib"
        reg_path = MODELS_DIR / "regression" / "best_regression_model.joblib"

        self.assertTrue(prep_path.exists(), "Preprocessor artifact must exist")
        self.assertTrue(clf_path.exists(), "Champion classification model must exist")
        self.assertTrue(reg_path.exists(), "Champion regression model must exist")

        prep = joblib.load(prep_path)
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)

        feat_df = engineer_financial_features(self.df_sample)
        X_trans = prep.transform(feat_df)
        self.assertGreater(X_trans.shape[1], 15)

        # Predict
        clf_pred = clf.predict(X_trans)
        reg_pred = reg.predict(X_trans)

        self.assertIn(clf_pred[0], [0, 1, 2])
        self.assertGreaterEqual(reg_pred[0], 500.0)
        self.assertLessEqual(reg_pred[0], 50000.0)

    def test_approval_advisor_counterfactual(self):
        prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
        clf = joblib.load(MODELS_DIR / "classification" / "best_classification_model.joblib")
        reg = joblib.load(MODELS_DIR / "regression" / "best_regression_model.joblib")
        advisor = ApprovalAdvisor(clf, reg, prep)

        # High risk stressed profile
        stressed = self.sample_row.copy()
        stressed["monthly_salary"] = 25000
        stressed["credit_score"] = 540
        stressed["requested_amount"] = 500000
        stressed["requested_tenure"] = 12

        res = advisor.predict_profile(pd.DataFrame([stressed]))
        self.assertIn(res["eligibility"], ["Not_Eligible", "High_Risk"])

        drivers = advisor.compute_risk_factors(stressed, res)
        self.assertGreater(len(drivers), 0)

        pathways = advisor.find_approval_pathways(stressed)
        self.assertGreater(len(pathways), 0)

    def test_crud_operations(self):
        test_rec = self.sample_row.copy()
        test_rec["applicant_name"] = "Unit Test Borrower"
        test_rec["predicted_eligibility"] = "Eligible"
        test_rec["predicted_max_emi"] = 18000.0
        test_rec["application_status"] = "Pending"

        # Create
        new_id = ApplicantService.create(test_rec)
        self.assertIsNotNone(new_id)

        # Read
        fetched = ApplicantService.get_by_id(new_id)
        self.assertEqual(fetched["applicant_name"], "Unit Test Borrower")

        # Update
        updated = ApplicantService.update(new_id, {"application_status": "Approved"})
        self.assertTrue(updated)
        self.assertEqual(ApplicantService.get_by_id(new_id)["application_status"], "Approved")

        # Delete
        deleted = ApplicantService.delete(new_id)
        self.assertTrue(deleted)
        self.assertIsNone(ApplicantService.get_by_id(new_id))

if __name__ == "__main__":
    unittest.main()
