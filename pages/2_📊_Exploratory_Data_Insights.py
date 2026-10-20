"""
Exploratory Data Analysis & Business Insights Dashboard for EMIPredict AI.
Visualizes credit distribution patterns, risk factor correlations, and scenario benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, EMI_SCENARIOS
from src.preprocessing import DataQualityAssessor

st.set_page_config(page_title="Data Insights & Analytics | EMIPredict AI", page_icon="📊", layout="wide")

# Check for Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.title("📊 Exploratory Data Analysis & Portfolio Insights")
st.markdown("Macro and granular analysis across 400,000 financial records & 5 lending scenarios.")

# Data Loading with caching
@st.cache_data
def load_eda_dataset():
    sample_csv = RAW_DATA_DIR / "emi_dataset_sample_50k.csv"
    full_csv = RAW_DATA_DIR / "emi_dataset_400k.csv"
    parquet_path = RAW_DATA_DIR / "emi_dataset_400k.parquet"

    if sample_csv.exists():
        return pd.read_csv(sample_csv)
    elif parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path).sample(n=30000, random_state=42)
        except Exception:
            pass
    elif full_csv.exists():
        return pd.read_csv(full_csv, nrows=30000)

    # If neither exists, generate on the fly
    from src.data_generator import generate_emi_dataset
    df = generate_emi_dataset(total_records_per_scenario=6000)
    return df

df = load_eda_dataset()

# Top Filters
st.markdown("### 🔍 Interactive Portfolio Filters")
f1, f2, f3, f4 = st.columns(4)

scenario_filter = f1.selectbox("Filter EMI Scenario:", ["All"] + list(EMI_SCENARIOS.keys()))
eligibility_filter = f2.multiselect("Eligibility Status:", ["Eligible", "High_Risk", "Not_Eligible"], default=["Eligible", "High_Risk", "Not_Eligible"])
salary_range = f3.slider("Monthly Gross Salary (₹):", min_value=15000, max_value=200000, value=(15000, 200000), step=5000)
age_range = f4.slider("Borrower Age (Years):", min_value=25, max_value=60, value=(25, 60))

# Apply filtering
filtered = df.copy()
if scenario_filter != "All":
    filtered = filtered[filtered["emi_scenario"] == scenario_filter]
if eligibility_filter:
    filtered = filtered[filtered["emi_eligibility"].isin(eligibility_filter)]
filtered = filtered[
    (filtered["monthly_salary"] >= salary_range[0]) & 
    (filtered["monthly_salary"] <= salary_range[1]) &
    (filtered["age"] >= age_range[0]) &
    (filtered["age"] <= age_range[1])
]

# Quick KPIs
k1, k2, k3, k4, k5 = st.columns(5)
total_sub = len(filtered)
elig_count = (filtered["emi_eligibility"] == "Eligible").sum()
high_risk_count = (filtered["emi_eligibility"] == "High_Risk").sum()
not_elig_count = (filtered["emi_eligibility"] == "Not_Eligible").sum()

k1.metric("Selected Cohort", f"{total_sub:,} records")
k2.metric("Approval Rate", f"{(elig_count / max(1, total_sub))*100:.1f}%", f"{elig_count:,} approved")
k3.metric("High Risk Proportion", f"{(high_risk_count / max(1, total_sub))*100:.1f}%", f"{high_risk_count:,} marginal")
k4.metric("Rejection Rate", f"{(not_elig_count / max(1, total_sub))*100:.1f}%", f"{not_elig_count:,} rejected", delta_color="inverse")
k5.metric("Median Safe EMI", f"₹{filtered['max_monthly_emi'].median():,.0f}")

st.markdown("---")

# Visual Analytics Section
tab1, tab2, tab3 = st.tabs(["📈 Scenario & Risk Analysis", "💳 Financial Correlates", "📋 Data Quality Audit"])

with tab1:
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### Scenario Volume & Underwriting Outcomes")
        scen_agg = df.groupby(["emi_scenario", "emi_eligibility"]).size().reset_index(name="count")
        
        if PLOTLY_AVAILABLE:
            fig1 = px.bar(
                scen_agg,
                x="emi_scenario",
                y="count",
                color="emi_eligibility",
                barmode="group",
                color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
                title="Eligibility Distribution Across Lending Scenarios"
            )
            fig1.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.bar_chart(df["emi_scenario"].value_counts())

    with c_chart2:
        st.markdown("#### Credit Score vs. Risk Categorization")
        if PLOTLY_AVAILABLE:
            fig2 = px.histogram(
                filtered,
                x="credit_score",
                color="emi_eligibility",
                nbins=40,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
                title="Credit Bureau Score Density by Decision Class"
            )
            fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(filtered.groupby("credit_score")["monthly_salary"].mean())

with tab2:
    c_chart3, c_chart4 = st.columns(2)

    with c_chart3:
        st.markdown("#### Monthly Salary vs. Max Safe Monthly EMI")
        if PLOTLY_AVAILABLE:
            sample_scatter = filtered.sample(n=min(2000, len(filtered)), random_state=42)
            fig3 = px.scatter(
                sample_scatter,
                x="monthly_salary",
                y="max_monthly_emi",
                color="emi_eligibility",
                opacity=0.6,
                color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
                trendline="ols",
                title="Borrower Salary vs. Algorithmic Max Safe EMI Cap"
            )
            fig3.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.scatter_chart(filtered[["monthly_salary", "max_monthly_emi"]].head(1000))

    with c_chart4:
        st.markdown("#### Employment Type & Stability Patterns")
        emp_agg = filtered.groupby(["employment_type", "emi_eligibility"]).size().unstack(fill_value=0)
        emp_agg_pct = emp_agg.div(emp_agg.sum(axis=1), axis=0) * 100
        
        if PLOTLY_AVAILABLE:
            fig4 = px.bar(
                emp_agg_pct.reset_index(),
                x="employment_type",
                y=["Eligible", "High_Risk", "Not_Eligible"],
                title="Approval Share by Employment Sector (%)",
                color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"}
            )
            fig4.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.dataframe(emp_agg_pct)

with tab3:
    st.markdown("#### Automated Banking Data Quality Audit")
    report = DataQualityAssessor.audit_dataset(df)
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Records Audited", f"{report['total_records']:,}")
    q2.metric("Missing Values", f"{report['missing_values_total']}")
    q3.metric("Duplicate Rows", f"{report['duplicate_records']}")
    q4.metric("Negative Salaries", f"{report['anomalies']['negative_salaries']}")

    st.markdown("##### Statistical Summary of 22 Core Variables")
    st.dataframe(df.describe().T.style.format("{:,.2f}"), use_container_width=True)
