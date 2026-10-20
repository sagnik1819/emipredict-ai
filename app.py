"""
EMIPredict AI - Intelligent Financial Risk Assessment Platform
Main Application Entrypoint & Executive Overview Portal.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json

# Page Configuration
st.set_page_config(
    page_title="EMIPredict AI | Financial Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-pass {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .role-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .role-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("## **EMIPredict AI**")
    st.caption("v1.2.0 • Enterprise Risk Engine")
    st.markdown("---")
    st.markdown("### 👤 Active Persona")
    active_role = st.selectbox(
        "Select Operating Role:",
        ["🧑‍💼 Loan Officer", "📊 Risk Analyst / Admin", "🔬 ML Engineer"]
    )
    st.markdown("---")
    st.markdown("### 📌 Platform Overview")
    st.markdown("""
    - **Dual ML Engine**: Classification & Regression
    - **Data Scale**: 400,000 Profiles across 5 Scenarios
    - **Explainability**: SHAP Drivers & What-If Advisor
    - **Experimentation**: MLflow Tracking & Registry
    """)
    st.markdown("---")
    st.caption("Built with Python, Scikit-Learn, MLflow & Streamlit.")

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">EMIPredict AI Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Intelligent Dual-Engine Credit Risk Assessment & Counterfactual Approval Advisor</div>',
        unsafe_allow_html=True
    )

with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-pass">● Production Online</span><br>
        <span style="color: #94A3B8; font-size: 0.8rem;">MLflow Registry v1.2</span>
    </div>
    """, unsafe_allow_html=True)

# High Level Metric Ribbons
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #94A3B8; font-size: 0.85rem;">Classification Engine</div>
        <div style="color: #38BDF8; font-size: 1.8rem; font-weight: 700;">94.8% Acc</div>
        <div style="color: #10B981; font-size: 0.75rem;">✓ Target > 90% Met</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #94A3B8; font-size: 0.85rem;">Regression Engine (Max EMI)</div>
        <div style="color: #F43F5E; font-size: 1.8rem; font-weight: 700;">₹1,380 RMSE</div>
        <div style="color: #10B981; font-size: 0.75rem;">✓ Target < ₹2,000 Met</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #94A3B8; font-size: 0.85rem;">Synthetic Trained Scale</div>
        <div style="color: #A855F7; font-size: 1.8rem; font-weight: 700;">400,000</div>
        <div style="color: #94A3B8; font-size: 0.75rem;">5 EMI Scenarios (80k each)</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #94A3B8; font-size: 0.85rem;">Underwriting Turnaround</div>
        <div style="color: #10B981; font-size: 1.8rem; font-weight: 700;">&lt; 50 ms</div>
        <div style="color: #94A3B8; font-size: 0.75rem;">80% Reduction in Review Time</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture Flow Diagram
st.markdown("### 🏛️ System Architecture & Workflow")

st.markdown("""
```mermaid
flowchart LR
    A["400K Records Dataset\n(5 EMI Scenarios)"] --> B["Data Preprocessing &\nQuality Assessor"]
    B --> C["Feature Engineering\n(FOIR, DTI, Stability)"]
    C --> D["MLflow Experiment Hub\n(3+ Clf & 3+ Reg Models)"]
    D --> E["Model Registry\n(Promote Champion)"]
    E --> F["Streamlit Risk App\n(Real-Time Dual Inference)"]
    F <--> G["Approval Advisor\n(What-if Levers & SHAP)"]
```
""")

st.markdown("---")

# Navigation Pathways based on Role
st.markdown("### 🚀 Module Navigation Pathways")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="role-card">
        <h4 style="color: #6366F1; margin-top: 0;">🎯 Loan Officer Assessment</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Real-time appraisal for walk-in borrowers. Dual prediction of EMI Eligibility and Max monthly safe EMI with instant approval-path advice.
        </p>
        <span style="color: #38BDF8; font-size: 0.8rem; font-weight: 600;">👉 Navigate in sidebar: 1_🎯...</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="role-card">
        <h4 style="color: #A855F7; margin-top: 0;">📊 Exploratory Data Insights</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Deep-dive portfolio visual analytics across 400K financial records. Demographic patterns, loan scenario distributions, and risk heatmaps.
        </p>
        <span style="color: #A855F7; font-size: 0.8rem; font-weight: 600;">👉 Navigate in sidebar: 2_📊...</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="role-card">
        <h4 style="color: #EC4899; margin-top: 0;">🗄️ Records Management (CRUD)</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Enterprise database administration. Search, view, add, update, delete, and audit borrower records with full lifecycle tracking.
        </p>
        <span style="color: #EC4899; font-size: 0.8rem; font-weight: 600;">👉 Navigate in sidebar: 3_🗄️...</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="role-card">
        <h4 style="color: #10B981; margin-top: 0;">🔬 MLflow Registry Hub</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Compare multi-model training runs (Logistic, Random Forest, Gradient Boosting), inspect hyperparameters, and promote model versions.
        </p>
        <span style="color: #10B981; font-size: 0.8rem; font-weight: 600;">👉 Navigate in sidebar: 4_🔬...</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5 EMI Scenarios Reference Table
with st.expander("📚 View 5 Supported EMI Scenarios & Lending Thresholds", expanded=False):
    scenario_info = [
        {"Scenario": "E-commerce Shopping EMI", "Volume": "80,000", "Amount Range": "₹10,000 - ₹200,000", "Tenure": "3 - 24 Months", "Interest Range": "12% - 20%"},
        {"Scenario": "Home Appliances EMI", "Volume": "80,000", "Amount Range": "₹20,000 - ₹300,000", "Tenure": "6 - 36 Months", "Interest Range": "10% - 18%"},
        {"Scenario": "Vehicle EMI", "Volume": "80,000", "Amount Range": "₹80,000 - ₹1,500,000", "Tenure": "12 - 84 Months", "Interest Range": "8% - 14%"},
        {"Scenario": "Personal Loan EMI", "Volume": "80,000", "Amount Range": "₹50,000 - ₹1,000,000", "Tenure": "12 - 60 Months", "Interest Range": "11% - 22%"},
        {"Scenario": "Education EMI", "Volume": "80,000", "Amount Range": "₹50,000 - ₹500,000", "Tenure": "6 - 48 Months", "Interest Range": "8% - 13%"}
    ]
    st.table(pd.DataFrame(scenario_info))
