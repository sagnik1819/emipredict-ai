"""
Smart Approval Advisor & Counterfactual What-If Engine for EMIPredict AI.
Implements automated counterfactual search to discover the minimal realistic adjustments
(tenure increase, downpayment/amount reduction, debt consolidation) to flip a High-Risk
or Not-Eligible loan application into 'Eligible', along with SHAP-inspired risk drivers.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .config import EMI_SCENARIOS
from .feature_engineering import engineer_financial_features

class ApprovalAdvisor:
    """
    Simulates counterfactual loan adjustments and computes risk driver attributions.
    Matches the exact sequence diagram flow:
    Streamlit App -> Advisor -> ML Service (Re-score variants) -> Smallest fix + SHAP
    """

    def __init__(self, classification_model, regression_model, preprocessor):
        self.clf = classification_model
        self.reg = regression_model
        self.preprocessor = preprocessor

    def predict_profile(self, profile_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs preprocessor and dual-model inference on a profile DataFrame."""
        feat_df = engineer_financial_features(profile_df)
        X_trans = self.preprocessor.transform(feat_df)

        # Classification prediction & probabilities
        clf_pred_code = self.clf.predict(X_trans)[0]
        clf_label = self.preprocessor.inv_class_mapping.get(clf_pred_code, "Eligible")

        probs = {}
        if hasattr(self.clf, "predict_proba"):
            raw_probs = self.clf.predict_proba(X_trans)[0]
            for code, name in self.preprocessor.inv_class_mapping.items():
                if code < len(raw_probs):
                    probs[name] = round(float(raw_probs[code]), 4)

        # Regression prediction
        reg_pred = float(self.reg.predict(X_trans)[0])
        max_safe_emi = max(500.0, min(50000.0, round(reg_pred, 2)))

        req_emi = float(feat_df["estimated_req_emi"].iloc[0])
        foir = float(feat_df["fixed_obligation_ratio"].iloc[0])
        dti = float(feat_df["debt_to_income_ratio"].iloc[0])
        disposable = float(feat_df["disposable_income"].iloc[0])

        return {
            "eligibility": clf_label,
            "probabilities": probs,
            "max_safe_emi": max_safe_emi,
            "estimated_req_emi": req_emi,
            "foir": foir,
            "dti": dti,
            "disposable_income": disposable,
            "feature_row": feat_df.iloc[0].to_dict()
        }

    def compute_risk_factors(self, profile_dict: Dict[str, Any], initial_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Derives local risk factor contributions (driver analysis) highlighting
        the specific financial bottlenecks preventing clean approval.
        """
        drivers = []
        monthly_salary = float(profile_dict.get("monthly_salary", 50000))
        credit_score = int(profile_dict.get("credit_score", 650))
        estimated_req_emi = initial_result["estimated_req_emi"]
        max_safe_emi = initial_result["max_safe_emi"]
        foir = initial_result["foir"]
        emergency_fund = float(profile_dict.get("emergency_fund", 0))
        expenses = float(initial_result["feature_row"].get("total_monthly_expenses", 25000))

        # 1. EMI Affordability Gap
        if estimated_req_emi > max_safe_emi:
            gap = estimated_req_emi - max_safe_emi
            pct = (gap / max_safe_emi) * 100
            drivers.append({
                "factor": "EMI Affordability Overrun",
                "severity": "Critical" if pct > 30 else "Moderate",
                "current_val": f"₹{estimated_req_emi:,.0f}/mo",
                "target_val": f"≤ ₹{max_safe_emi:,.0f}/mo",
                "impact": f"Requested installment exceeds algorithmic safe capacity by ₹{gap:,.0f} (+{pct:.1f}%)."
            })

        # 2. Fixed Obligation to Income (FOIR)
        if foir > 0.50:
            drivers.append({
                "factor": "High Fixed Obligation Ratio (FOIR)",
                "severity": "Critical" if foir > 0.65 else "Moderate",
                "current_val": f"{foir*100:.1f}%",
                "target_val": "≤ 50.0%",
                "impact": f"Combined debt + rent obligations consume {foir*100:.1f}% of gross income, exceeding the 50% prudent banking ceiling."
            })

        # 3. Credit Score Benchmark
        if credit_score < 680:
            drivers.append({
                "factor": "Sub-Prime Credit Score",
                "severity": "Critical" if credit_score < 600 else "Moderate",
                "current_val": f"{credit_score}",
                "target_val": "≥ 680",
                "impact": f"Credit bureau score places applicant in higher default risk cohort."
            })

        # 4. Emergency Cushion
        if expenses > 0 and (emergency_fund / expenses) < 1.0:
            cushion = emergency_fund / expenses
            drivers.append({
                "factor": "Insufficient Emergency Liquidity",
                "severity": "Low",
                "current_val": f"{cushion:.1f} months",
                "target_val": "≥ 1.0 month",
                "impact": f"Savings provide less than 30 days of living expenses in case of unexpected income shock."
            })

        if not drivers:
            drivers.append({
                "factor": "Marginal Risk Buffer",
                "severity": "Low",
                "current_val": "Borderline",
                "target_val": "Standard",
                "impact": "Applicant meets minimum hurdles but overall risk-reward requires slight rate surcharge or downpayment."
            })

        return drivers

    def find_approval_pathways(self, profile_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Counterfactual search across realistic levers:
        1. Tenure extension
        2. Loan amount reduction
        3. Debt consolidation (existing EMI paydown)
        4. Combined optimal package
        """
        scenario = profile_dict.get("emi_scenario", "Personal Loan EMI")
        scenario_spec = EMI_SCENARIOS.get(scenario, EMI_SCENARIOS["Personal Loan EMI"])
        max_tenure = scenario_spec["max_tenure"]
        cur_tenure = int(profile_dict.get("requested_tenure", 12))
        cur_amount = float(profile_dict.get("requested_amount", 100000))
        cur_existing_emi = float(profile_dict.get("current_emi_amount", 0))

        pathways = []

        # Strategy 1: Tenure Extension (if not already at max)
        if cur_tenure < max_tenure:
            test_tenures = [t for t in [18, 24, 36, 48, 60, 84] if t > cur_tenure and t <= max_tenure]
            for t in test_tenures:
                var_dict = profile_dict.copy()
                var_dict["requested_tenure"] = t
                res = self.predict_profile(pd.DataFrame([var_dict]))
                if res["eligibility"] == "Eligible":
                    pathways.append({
                        "strategy": "Extend Repayment Tenure",
                        "description": f"Increase repayment term from {cur_tenure} months to {t} months.",
                        "new_emi": f"₹{res['estimated_req_emi']:,.0f}/mo",
                        "savings_per_month": f"₹{(profile_dict.get('requested_amount', 0)/cur_tenure - res['estimated_req_emi']):,.0f}",
                        "outcome": "Approved (Eligible)",
                        "difficulty": "Easy (No upfront capital required)"
                    })
                    break

        # Strategy 2: Loan Amount Reduction / Higher Downpayment
        for cut_pct in [10, 15, 20, 25, 30, 40]:
            new_amt = round(cur_amount * (1.0 - cut_pct / 100.0), -3)
            var_dict = profile_dict.copy()
            var_dict["requested_amount"] = new_amt
            res = self.predict_profile(pd.DataFrame([var_dict]))
            if res["eligibility"] == "Eligible":
                pathways.append({
                    "strategy": f"Reduce Principal by {cut_pct}% (Downpayment)",
                    "description": f"Lower requested loan from ₹{cur_amount:,.0f} to ₹{new_amt:,.0f} (Borrower provides ₹{cur_amount - new_amt:,.0f} margin money).",
                    "new_emi": f"₹{res['estimated_req_emi']:,.0f}/mo",
                    "savings_per_month": f"₹{(profile_dict.get('requested_amount', 0)/cur_tenure - res['estimated_req_emi']):,.0f}",
                    "outcome": "Approved (Eligible)",
                    "difficulty": "Moderate (Requires self-contribution)"
                })
                break

        # Strategy 3: Existing Debt Settle / Payoff
        if cur_existing_emi > 0:
            for paydown_pct in [50, 100]:
                var_dict = profile_dict.copy()
                var_dict["current_emi_amount"] = cur_existing_emi * (1.0 - paydown_pct / 100.0)
                if paydown_pct == 100:
                    var_dict["existing_loans"] = 0
                res = self.predict_profile(pd.DataFrame([var_dict]))
                if res["eligibility"] == "Eligible":
                    label = "Full Debt Clearance" if paydown_pct == 100 else "Partial Debt Consolidation (50%)"
                    pathways.append({
                        "strategy": label,
                        "description": f"Close existing ongoing loan obligations (reduces existing EMI burden by ₹{cur_existing_emi * (paydown_pct/100):,.0f}/mo).",
                        "new_emi": f"₹{res['estimated_req_emi']:,.0f}/mo",
                        "savings_per_month": "N/A",
                        "outcome": "Approved (Eligible)",
                        "difficulty": "Moderate (Depends on borrower liquidity)"
                    })
                    break

        # Strategy 4: Co-Borrower / Guarantor Fallback
        if not pathways:
            pathways.append({
                "strategy": "Add Creditworthy Co-Applicant / Guarantor",
                "description": "Combine household incomes with a primary earning co-borrower to raise joint debt servicing capacity.",
                "new_emi": f"~₹{cur_amount / cur_tenure:,.0f}/mo",
                "savings_per_month": "N/A",
                "outcome": "Eligible via Joint Underwriting",
                "difficulty": "Subject to secondary borrower appraisal"
            })

        return pathways
