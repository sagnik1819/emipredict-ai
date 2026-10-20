"""
Applicant Records Management (CRUD) Dashboard for EMIPredict AI.
Provides complete Create, Read, Update, and Delete operations for loan applicants,
with status auditing, search, filtering, and CSV export.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.crud_service import ApplicantService, init_db
from src.config import EMI_SCENARIOS, RAW_DATA_DIR

st.set_page_config(page_title="Applicant Records CRUD | EMIPredict AI", page_icon="🗄️", layout="wide")

st.title("🗄️ Borrower Records Management (CRUD)")
st.markdown("Administrative underwriting console for creating, inspecting, updating, and auditing borrower profiles.")

init_db()

# Seed if empty
sample_csv = RAW_DATA_DIR / "emi_dataset_sample_50k.csv"
if sample_csv.exists():
    try:
        sample_df = pd.read_csv(sample_csv, nrows=100)
        ApplicantService.seed_initial_records(sample_df)
    except Exception as e:
        pass

# Top Portfolio Stats
summary = ApplicantService.get_portfolio_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Managed Records", f"{summary['total_records']:,}")
c2.metric("Average Requested Loan", f"₹{summary['avg_loan_requested']:,.0f}")
c3.metric("Average Monthly Salary", f"₹{summary['avg_monthly_salary']:,.0f}")
c4.metric("Average Credit Score", f"{summary['avg_credit_score']}")

st.markdown("---")

# Filter and Search Bar
st.markdown("### 🔍 Query & Filter Applicants")
col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1])

search_query = col_s1.text_input("Search by Name or Application ID:", placeholder="e.g. Arjun or 12")
scenario_filter = col_s2.selectbox("Scenario Filter:", ["All"] + list(EMI_SCENARIOS.keys()))
status_filter = col_s3.selectbox("Status Filter:", ["All", "Approved", "Under Review", "Pending", "Flagged"])
records_limit = col_s4.selectbox("Limit Records:", [50, 100, 250, 500], index=1)

records_df = ApplicantService.list_all(
    search=search_query if search_query else None,
    scenario=scenario_filter,
    status=status_filter,
    limit=records_limit
)

# Tabbed Interface for CRUD
tab_view, tab_add, tab_edit, tab_delete = st.tabs([
    "📋 View Records Table",
    "➕ Register New Applicant (Create)",
    "✏️ Edit Profile & Status (Update)",
    "🗑️ Delete Record (Delete)"
])

# 1. READ / VIEW
with tab_view:
    st.markdown(f"**Displaying {len(records_df):,} matching applicant records:**")
    
    if not records_df.empty:
        display_cols = [
            "id", "applicant_name", "age", "monthly_salary", "credit_score",
            "emi_scenario", "requested_amount", "requested_tenure",
            "predicted_eligibility", "predicted_max_emi", "application_status", "updated_at"
        ]
        available_cols = [c for c in display_cols if c in records_df.columns]
        
        # Color coding status
        st.dataframe(
            records_df[available_cols].style.format({
                "monthly_salary": "₹{:,.0f}",
                "requested_amount": "₹{:,.0f}",
                "predicted_max_emi": "₹{:,.0f}"
            }),
            use_container_width=True,
            height=420
        )

        # Download CSV
        csv_data = records_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Records to CSV",
            data=csv_data,
            file_name=f"emi_applicants_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No records found matching the query criteria.")

# 2. CREATE
with tab_add:
    st.markdown("#### Register a New Borrower Profile")
    with st.form("create_applicant_form", clear_on_submit=True):
        ac1, ac2, ac3 = st.columns(3)
        new_name = ac1.text_input("Applicant Name*", value="Sanjay Patel")
        new_age = ac2.number_input("Age", min_value=25, max_value=60, value=35)
        new_gender = ac3.selectbox("Gender", ["Male", "Female"])

        ac4, ac5, ac6 = st.columns(3)
        new_marital = ac4.selectbox("Marital Status", ["Married", "Single"])
        new_edu = ac5.selectbox("Education", ["Graduate", "Post Graduate", "High School", "Professional"])
        new_salary = ac6.number_input("Monthly Gross Salary (₹)*", min_value=15000, max_value=500000, value=75000, step=5000)

        ac7, ac8, ac9 = st.columns(3)
        new_emp = ac7.selectbox("Employment Sector", ["Private", "Government", "Self-employed"])
        new_yoe = ac8.number_input("Years of Employment", min_value=0, max_value=35, value=7)
        new_company = ac9.selectbox("Company Type", ["Large MNC", "Mid-size", "Startup", "Government"])

        ac10, ac11, ac12 = st.columns(3)
        new_house = ac10.selectbox("Housing Type", ["Rented", "Own", "Family"])
        new_rent = ac11.number_input("Monthly Rent (₹)", min_value=0, max_value=80000, value=15000)
        new_cs = ac12.slider("Credit Bureau Score*", min_value=300, max_value=850, value=750)

        ac13, ac14, ac15 = st.columns(3)
        new_scenario = ac13.selectbox("Loan Scenario*", list(EMI_SCENARIOS.keys()))
        new_amount = ac14.number_input("Requested Loan Amount (₹)*", min_value=10000, max_value=2000000, value=300000, step=10000)
        new_tenure = ac15.number_input("Requested Tenure (Months)*", min_value=3, max_value=84, value=24)

        submit_btn = st.form_submit_button("➕ Save New Applicant")
        if submit_btn:
            new_record = {
                "applicant_name": new_name, "age": new_age, "gender": new_gender,
                "marital_status": new_marital, "education": new_edu,
                "monthly_salary": new_salary, "employment_type": new_emp,
                "years_of_employment": new_yoe, "company_type": new_company,
                "house_type": new_house, "monthly_rent": new_rent,
                "family_size": 3, "dependents": 1,
                "school_fees": 3000, "college_fees": 0, "travel_expenses": 3500,
                "groceries_utilities": 12000, "other_monthly_expenses": 2500,
                "existing_loans": 0, "current_emi_amount": 0,
                "credit_score": new_cs, "bank_balance": new_salary * 2,
                "emergency_fund": new_salary * 0.8,
                "emi_scenario": new_scenario, "requested_amount": new_amount,
                "requested_tenure": new_tenure,
                "predicted_eligibility": "Eligible" if new_cs >= 680 else "High_Risk",
                "predicted_max_emi": round(new_salary * 0.40, 2),
                "application_status": "Pending",
                "notes": "Added manually via CRUD interface"
            }
            new_id = ApplicantService.create(new_record)
            st.success(f"🎉 New Applicant '{new_name}' created successfully with Record ID #{new_id}!")
            st.rerun()

# 3. UPDATE
with tab_edit:
    st.markdown("#### Update Applicant Profile & Underwriting Decision")
    if not records_df.empty:
        selected_id = st.selectbox(
            "Select Applicant to Edit:",
            records_df["id"].tolist(),
            format_func=lambda x: f"ID #{x} - {records_df[records_df['id']==x]['applicant_name'].values[0]}"
        )
        curr = ApplicantService.get_by_id(selected_id)
        if curr:
            with st.form("edit_applicant_form"):
                eu1, eu2, eu3 = st.columns(3)
                up_name = eu1.text_input("Full Name", value=curr["applicant_name"])
                up_salary = eu2.number_input("Monthly Salary (₹)", value=int(curr["monthly_salary"]), step=2500)
                up_cs = eu3.slider("Credit Score", 300, 850, value=int(curr["credit_score"]))

                eu4, eu5, eu6 = st.columns(3)
                up_status = eu4.selectbox("Application Status", ["Approved", "Under Review", "Pending", "Flagged"],
                                          index=["Approved", "Under Review", "Pending", "Flagged"].index(curr.get("application_status", "Pending")))
                up_amount = eu5.number_input("Requested Amount (₹)", value=int(curr["requested_amount"]), step=10000)
                up_notes = eu6.text_input("Reviewer Notes", value=str(curr.get("notes", "")))

                up_submit = st.form_submit_button("💾 Save Profile Changes")
                if up_submit:
                    updates = {
                        "applicant_name": up_name,
                        "monthly_salary": up_salary,
                        "credit_score": up_cs,
                        "application_status": up_status,
                        "requested_amount": up_amount,
                        "notes": up_notes
                    }
                    ApplicantService.update(selected_id, updates)
                    st.success(f"Record #{selected_id} successfully updated!")
                    st.rerun()
    else:
        st.info("No records available to edit.")

# 4. DELETE
with tab_delete:
    st.markdown("#### Permanent Record Deletion")
    if not records_df.empty:
        del_id = st.selectbox(
            "Select Applicant Record to Remove:",
            records_df["id"].tolist(),
            format_func=lambda x: f"ID #{x} - {records_df[records_df['id']==x]['applicant_name'].values[0]}"
        )
        st.warning(f"⚠️ Are you sure you want to permanently delete Record #{del_id}? This operation cannot be reversed.")
        if st.button("🚨 Confirm and Delete Record", type="primary"):
            ApplicantService.delete(del_id)
            st.success(f"Record #{del_id} successfully purged from database.")
            st.rerun()
    else:
        st.info("No records available to delete.")
