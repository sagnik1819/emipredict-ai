"""
Loan Officer Risk Assessment & Real-Time Underwriting Dashboard.
Provides real-time dual ML prediction (Eligibility + Max Monthly EMI),
risk factor explanations, and counterfactual approval pathway advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.config import (
    EMI_SCENARIOS,
    MODELS_DIR,
    RAW_DATA_DIR
)
from src.preprocessing import PreprocessingPipeline
from src.advisor_engine import ApprovalAdvisor
from src.crud_service import ApplicantService

st.set_page_config(page_title="Loan Officer Assessment | EMIPredict AI", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #818CF8;
        border-bottom: 2px solid #312E81;
        padding-bottom: 5px;
        margin-top: 15px;
        margin-bottom: 12px;
    }
    .status-badge-eligible {
        background: #064E3B;
        color: #34D399;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #10B981;
    }
    .status-badge-highrisk {
        background: #78350F;
        color: #FBBF24;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #F59E0B;
    }
    .status-badge-noteligible {
        background: #7F1D1D;
        color: #F87171;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #EF4444;
    }
    .card-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load models
@st.cache_resource
def load_production_pipeline():
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    clf_path = MODELS_DIR / "classification" / "best_classification_model.joblib"
    reg_path = MODELS_DIR / "regression" / "best_regression_model.joblib"

    if not preprocessor_path.exists() or not clf_path.exists() or not reg_path.exists():
        return None, None, None

    preprocessor = joblib.load(preprocessor_path)
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return preprocessor, clf, reg

preprocessor, clf, reg = load_production_pipeline()

# Title
st.title("🎯 Real-Time Loan Eligibility & Safe EMI Assessment")
st.markdown("Automated algorithmic underwriting engine evaluating 22 customer attributes against dual ML models.")

if preprocessor is None or clf is None or reg is None:
    st.warning("⚠️ ML models are currently compiling or not yet trained. Please visit the **4_🔬_MLflow_Experiment_Registry** page to initialize model training.")
    st.stop()

advisor = ApprovalAdvisor(clf, reg, preprocessor)

# Quick Presets for Demo
st.markdown("### ⚡ Quick Profile Presets")
preset_col1, preset_col2, preset_col3 = st.columns(3)

if "profile_data" not in st.session_state:
    st.session_state["profile_data"] = {
        "name": "Arjun Sharma", "age": 34, "gender": "Male", "marital": "Married", "edu": "Graduate",
        "salary": 65000, "emp_type": "Private", "yoe": 6, "comp_type": "Large MNC",
        "house": "Rented", "rent": 14000, "fam_size": 3, "dep": 1,
        "school": 4500, "college": 0, "travel": 3500, "groceries": 12000, "other": 3000,
        "loans": 1, "cur_emi": 5500, "cs": 740, "balance": 120000, "emerg": 60000,
        "scenario": "Vehicle EMI", "amount": 450000, "tenure": 36
    }

with preset_col1:
    if st.button("🟢 Load Prime Borrower (Eligible)"):
        st.session_state["profile_data"] = {
            "name": "Priya Sen", "age": 32, "gender": "Female", "marital": "Married", "edu": "Post Graduate",
            "salary": 95000, "emp_type": "Private", "yoe": 8, "comp_type": "Large MNC",
            "house": "Own", "rent": 0, "fam_size": 2, "dep": 0,
            "school": 0, "college": 0, "travel": 4000, "groceries": 14000, "other": 4000,
            "loans": 0, "cur_emi": 0, "cs": 785, "balance": 240000, "emerg": 150000,
            "scenario": "Home Appliances EMI", "amount": 120000, "tenure": 24
        }
        st.rerun()

with preset_col2:
    if st.button("🟡 Load Marginal Borrower (High Risk)"):
        st.session_state["profile_data"] = {
            "name": "Vikram Patel", "age": 28, "gender": "Male", "marital": "Single", "edu": "Graduate",
            "salary": 42000, "emp_type": "Private", "yoe": 3, "comp_type": "Startup",
            "house": "Rented", "rent": 12000, "fam_size": 1, "dep": 0,
            "school": 0, "college": 0, "travel": 3000, "groceries": 8000, "other": 2500,
            "loans": 1, "cur_emi": 6000, "cs": 645, "balance": 35000, "emerg": 15000,
            "scenario": "Personal Loan EMI", "amount": 300000, "tenure": 24
        }
        st.rerun()

with preset_col3:
    if st.button("🔴 Load Strained Borrower (Not Eligible)"):
        st.session_state["profile_data"] = {
            "name": "Rohan Verma", "age": 36, "gender": "Male", "marital": "Married", "edu": "High School",
            "salary": 30000, "emp_type": "Self-employed", "yoe": 2, "comp_type": "Mid-size",
            "house": "Rented", "rent": 11000, "fam_size": 4, "dep": 2,
            "school": 5000, "college": 0, "travel": 2500, "groceries": 9000, "other": 2000,
            "loans": 2, "cur_emi": 8500, "cs": 540, "balance": 15000, "emerg": 5000,
            "scenario": "Vehicle EMI", "amount": 500000, "tenure": 36
        }
        st.rerun()

st.markdown("---")

cur_data = st.session_state["profile_data"]

# Input Form in Clean Two-Column Layout
left_form, right_form = st.columns([1, 1], gap="large")

with left_form:
    st.markdown('<div class="section-header">1. Personal & Employment Profile</div>', unsafe_allow_html=True)
    c_n1, c_n2 = st.columns([2, 1])
    applicant_name = c_n1.text_input("Applicant Full Name", value=cur_data.get("name", "Arjun Sharma"))
    age = c_n2.number_input("Age (Years)", min_value=25, max_value=60, value=int(cur_data.get("age", 34)))

    c_d1, c_d2, c_d3 = st.columns(3)
    gender = c_d1.selectbox("Gender", ["Male", "Female"], index=0 if cur_data.get("gender") == "Male" else 1)
    marital_status = c_d2.selectbox("Marital Status", ["Married", "Single"], index=0 if cur_data.get("marital") == "Married" else 1)
    education = c_d3.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"],
                               index=["High School", "Graduate", "Post Graduate", "Professional"].index(cur_data.get("edu", "Graduate")))

    c_e1, c_e2 = st.columns(2)
    monthly_salary = c_e1.number_input("Gross Monthly Salary (₹)", min_value=15000, max_value=500000, step=2500, value=int(cur_data.get("salary", 65000)))
    employment_type = c_e2.selectbox("Employment Type", ["Private", "Government", "Self-employed"],
                                      index=["Private", "Government", "Self-employed"].index(cur_data.get("emp_type", "Private")))

    c_e3, c_e4 = st.columns(2)
    years_of_employment = c_e3.slider("Years of Employment", min_value=0, max_value=35, value=int(cur_data.get("yoe", 6)))
    company_type = c_e4.selectbox("Company Type", ["Startup", "Mid-size", "Large MNC", "Government"],
                                  index=["Startup", "Mid-size", "Large MNC", "Government"].index(cur_data.get("comp_type", "Large MNC")))

    st.markdown('<div class="section-header">2. Housing & Living Obligations</div>', unsafe_allow_html=True)
    c_h1, c_h2 = st.columns(2)
    house_type = c_h1.selectbox("Housing Status", ["Rented", "Own", "Family"],
                                index=["Rented", "Own", "Family"].index(cur_data.get("house", "Rented")))
    monthly_rent = c_h2.number_input("Monthly Rent Expense (₹)", min_value=0, max_value=100000, step=1000,
                                     value=0 if house_type != "Rented" else int(cur_data.get("rent", 14000)))

    c_f1, c_f2 = st.columns(2)
    family_size = c_f1.number_input("Total Family Members", min_value=1, max_value=10, value=int(cur_data.get("fam_size", 3)))
    dependents = c_f2.number_input("Financial Dependents", min_value=0, max_value=7, value=int(cur_data.get("dep", 1)))

    c_x1, c_x2 = st.columns(2)
    school_fees = c_x1.number_input("School Fees (₹)", min_value=0, max_value=50000, step=500, value=int(cur_data.get("school", 4500)))
    college_fees = c_x2.number_input("College Fees (₹)", min_value=0, max_value=80000, step=1000, value=int(cur_data.get("college", 0)))

    c_x3, c_x4, c_x5 = st.columns(3)
    travel_expenses = c_x3.number_input("Travel (₹)", min_value=0, max_value=30000, step=500, value=int(cur_data.get("travel", 3500)))
    groceries_utilities = c_x4.number_input("Utilities & Food (₹)", min_value=1000, max_value=60000, step=1000, value=int(cur_data.get("groceries", 12000)))
    other_monthly_expenses = c_x5.number_input("Other Living (₹)", min_value=0, max_value=40000, step=500, value=int(cur_data.get("other", 3000)))

with right_form:
    st.markdown('<div class="section-header">3. Financial Status & Credit History</div>', unsafe_allow_html=True)
    c_b1, c_b2 = st.columns(2)
    existing_loans = c_b1.number_input("Existing Active Loans", min_value=0, max_value=5, value=int(cur_data.get("loans", 1)))
    current_emi_amount = c_b2.number_input("Ongoing Monthly EMI Burden (₹)", min_value=0, max_value=100000, step=500,
                                          value=0 if existing_loans == 0 else int(cur_data.get("cur_emi", 5500)))

    c_s1, c_s2, c_s3 = st.columns(3)
    credit_score = c_s1.slider("Credit Bureau Score", min_value=300, max_value=850, value=int(cur_data.get("cs", 740)))
    bank_balance = c_s2.number_input("Bank Balance (₹)", min_value=1000, max_value=2000000, step=10000, value=int(cur_data.get("balance", 120000)))
    emergency_fund = c_s3.number_input("Emergency Savings (₹)", min_value=0, max_value=1000000, step=5000, value=int(cur_data.get("emerg", 60000)))

    st.markdown('<div class="section-header">4. Loan Application Details</div>', unsafe_allow_html=True)
    scenario_list = list(EMI_SCENARIOS.keys())
    current_scen = cur_data.get("scenario", "Vehicle EMI")
    scen_idx = scenario_list.index(current_scen) if current_scen in scenario_list else 0
    emi_scenario = st.selectbox("Lending Category / EMI Scenario", scenario_list, index=scen_idx)

    # Dynamic limits based on scenario
    specs = EMI_SCENARIOS[emi_scenario]
    c_l1, c_l2 = st.columns(2)
    requested_amount = c_l1.slider(
        f"Loan Amount (₹) [Min: {specs['min_amount']:,} - Max: {specs['max_amount']:,}]",
        min_value=specs["min_amount"],
        max_value=specs["max_amount"],
        value=int(min(max(cur_data.get("amount", 200000), specs["min_amount"]), specs["max_amount"])),
        step=5000
    )
    requested_tenure = c_l2.slider(
        f"Repayment Tenure (Months) [Min: {specs['min_tenure']} - Max: {specs['max_tenure']}]",
        min_value=specs["min_tenure"],
        max_value=specs["max_tenure"],
        value=int(min(max(cur_data.get("tenure", 24), specs["min_tenure"]), specs["max_tenure"])),
        step=3 if specs["max_tenure"] > 12 else 1
    )

# Build applicant profile dictionary
profile = {
    "age": age, "gender": gender, "marital_status": marital_status, "education": education,
    "monthly_salary": monthly_salary, "employment_type": employment_type,
    "years_of_employment": years_of_employment, "company_type": company_type,
    "house_type": house_type, "monthly_rent": monthly_rent, "family_size": family_size,
    "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
    "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses, "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount, "credit_score": credit_score,
    "bank_balance": bank_balance, "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario, "requested_amount": requested_amount,
    "requested_tenure": requested_tenure
}

profile_df = pd.DataFrame([profile])

# Real-time Dual Prediction
result = advisor.predict_profile(profile_df)
eligibility = result["eligibility"]
max_safe_emi = result["max_safe_emi"]
req_emi = result["estimated_req_emi"]
foir = result["foir"]
dti = result["dti"]
disposable = result["disposable_income"]

st.markdown("---")
st.markdown("## 📋 Underwriting Evaluation Results")

col_badge, col_emi_metric, col_foir_metric, col_disp_metric = st.columns([1.2, 1, 1, 1])

with col_badge:
    st.markdown("**Eligibility Verdict**")
    if eligibility == "Eligible":
        st.markdown('<div class="status-badge-eligible">✅ ELIGIBLE</div>', unsafe_allow_html=True)
        st.caption("Low Risk Profile • Standard Interest Rate")
    elif eligibility == "High_Risk":
        st.markdown('<div class="status-badge-highrisk">⚠️ HIGH RISK</div>', unsafe_allow_html=True)
        st.caption("Marginal Case • Rate Surcharge / Collateral Needed")
    else:
        st.markdown('<div class="status-badge-noteligible">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
        st.caption("High Risk • Policy Threshold Breached")

with col_emi_metric:
    st.metric(
        label="Algorithmic Max Safe Monthly EMI",
        value=f"₹{max_safe_emi:,.0f}",
        delta=f"Requested: ₹{req_emi:,.0f}/mo",
        delta_color="normal" if req_emi <= max_safe_emi else "inverse"
    )

with col_foir_metric:
    st.metric(
        label="Fixed Obligation Ratio (FOIR)",
        value=f"{foir*100:.1f}%",
        delta="Ceiling: 50.0%",
        delta_color="normal" if foir <= 0.50 else "inverse"
    )

with col_disp_metric:
    st.metric(
        label="Net Unencumbered Cashflow",
        value=f"₹{disposable:,.0f}",
        delta=f"DTI: {dti*100:.1f}%"
    )

# Save to Database Section
st.markdown("<br>", unsafe_allow_html=True)
save_col1, save_col2 = st.columns([2, 1])
with save_col1:
    notes = st.text_input("Underwriter Notes / Memo:", value="Auto-evaluated via EMIPredict AI.")
with save_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("💾 Log & Save Application Record", use_container_width=True):
        rec_data = profile.copy()
        rec_data["applicant_name"] = applicant_name
        rec_data["predicted_eligibility"] = eligibility
        rec_data["predicted_max_emi"] = max_safe_emi
        rec_data["application_status"] = "Approved" if eligibility == "Eligible" else "Under Review"
        rec_data["notes"] = notes
        record_id = ApplicantService.create(rec_data)
        st.success(f"Application recorded successfully! Assigned ID: #{record_id}")

# -------------------------------------------------------------
# SMART APPROVAL ADVISOR (What-If Counterfactuals & SHAP Drivers)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧭 Smart Approval Advisor & Approval-Path Insights")

if eligibility == "Eligible":
    st.success("🎉 **Application is fully qualified!** No corrective restructuring required. The borrower can proceed directly to document verification and loan agreement disbursement.")
else:
    st.info("💡 **Borrower qualification gap detected.** Below is the automated counterfactual restructuring plan to flip this loan application to **'Eligible'**.")

    col_drivers, col_remedy = st.columns([1, 1], gap="large")

    with col_drivers:
        st.markdown("### 🔍 Key Bottlenecks / Risk Drivers")
        drivers = advisor.compute_risk_factors(profile, result)
        for d in drivers:
            st.markdown(f"""
            <div class="card-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #F8FAFC;">{d['factor']}</strong>
                    <span style="color: {'#EF4444' if d['severity'] == 'Critical' else '#F59E0B'}; font-size: 0.8rem; font-weight: 700;">
                        ● {d['severity']}
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 5px;">
                    Current: <code>{d['current_val']}</code> → Target: <code>{d['target_val']}</code>
                </div>
                <div style="color: #CBD5E1; font-size: 0.82rem; margin-top: 6px;">
                    {d['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_remedy:
        st.markdown("### 🛠️ Actionable Restructuring Pathways")
        pathways = advisor.find_approval_pathways(profile)

        for idx, p in enumerate(pathways):
            with st.container():
                st.markdown(f"""
                <div class="card-box" style="border-left: 4px solid #10B981;">
                    <div style="display: flex; justify-content: space-between;">
                        <h4 style="color: #34D399; margin: 0;">Option {chr(65+idx)}: {p['strategy']}</h4>
                        <span style="color: #A7F3D0; font-size: 0.8rem;">{p['difficulty']}</span>
                    </div>
                    <p style="color: #E2E8F0; font-size: 0.88rem; margin: 8px 0;">
                        {p['description']}
                    </p>
                    <div style="display: flex; gap: 20px; font-size: 0.82rem; color: #94A3B8;">
                        <span>New Installment: <b style="color: #38BDF8;">{p['new_emi']}</b></span>
                        <span>Outcome: <b style="color: #10B981;">{p['outcome']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
