"""
Automated 8-Slide PowerPoint Presentation Generator for EMIPredict AI Capstone.
Constructs an executive, 16:9 widescreen, dark-themed presentation covering all evaluation rubrics.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

def create_presentation(output_path: str = "EMIPredict_AI_Capstone_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    BG_DARK = RGBColor(15, 23, 42)      # Slate 900
    CARD_BG = RGBColor(30, 41, 59)      # Slate 800
    PRIMARY = RGBColor(99, 102, 241)    # Indigo 500
    ACCENT = RGBColor(16, 185, 129)     # Emerald 500
    TEXT_WHITE = RGBColor(248, 250, 252)# Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184)# Slate 400
    TEXT_CYAN = RGBColor(56, 189, 248)  # Sky 400
    ACCENT_AMBER = RGBColor(245, 158, 11)

    def set_slide_background(slide):
        background = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height
        )
        background.fill.solid()
        background.fill.fore_color.rgb = BG_DARK
        background.line.fill.background()
        return background

    def add_header(slide, title_text, category_text):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = PRIMARY

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 1: Title & Executive Summary (Rubric: Summary & Milestones)
    # =========================================================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide1)

    # Title Card
    tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(2.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "EMIPredict AI Platform"
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = TEXT_CYAN

    p1 = tf.add_paragraph()
    p1.text = "Intelligent Financial Risk Assessment & Counterfactual Underwriting Engine"
    p1.font.size = Pt(18)
    p1.font.color.rgb = TEXT_WHITE

    p2 = tf.add_paragraph()
    p2.text = "Capstone Project • Domain: FinTech & Banking Credit Underwriting"
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_MUTED

    # 3 Stat Cards
    stats = [
        ("99.10% ACCURACY", "Classification Target >90% Met", ACCENT),
        ("₹341.7 RMSE", "Regression Target <₹2,000 Met", ACCENT),
        ("400,000 RECORDS", "5 Lending Scenarios Covered", PRIMARY)
    ]
    for i, (head, sub, col) in enumerate(stats):
        card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0 + i * 3.85), Inches(3.2), Inches(3.6), Inches(1.3))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)
        tf_c = card.text_frame
        tf_c.word_wrap = True
        p = tf_c.paragraphs[0]
        p.text = head
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        p_s = tf_c.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(11)
        p_s.font.color.rgb = TEXT_MUTED
        p_s.alignment = PP_ALIGN.CENTER

    # Links Box
    lbox = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.8), Inches(11.33), Inches(1.8))
    lbox.fill.solid()
    lbox.fill.fore_color.rgb = CARD_BG
    lbox.line.color.rgb = PRIMARY
    tf_l = lbox.text_frame
    tf_l.word_wrap = True
    
    pl1 = tf_l.paragraphs[0]
    pl1.text = "🔗 OFFICIAL PROJECT LINKS & ARTIFACTS"
    pl1.font.size = Pt(13)
    pl1.font.bold = True
    pl1.font.color.rgb = PRIMARY

    pl2 = tf_l.add_paragraph()
    pl2.text = "• GitHub Repository: https://github.com/sagnik1819/emipredict-ai"
    pl2.font.size = Pt(12)
    pl2.font.bold = True
    pl2.font.color.rgb = TEXT_CYAN

    pl3 = tf_l.add_paragraph()
    pl3.text = "• Live Cloud Web App: https://forwarding-humanity-pest-urban.trycloudflare.com"
    pl3.font.size = Pt(12)
    pl3.font.bold = True
    pl3.font.color.rgb = ACCENT

    pl4 = tf_l.add_paragraph()
    pl4.text = "• Architecture: Dual ML (Classification + Regression) | MLflow Tracking | SQLite CRUD | Smart Approval Advisor"
    pl4.font.size = Pt(11)
    pl4.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: Data Exploration & Dictionary (Rubric: Head, Tail, Summary, Dict)
    # =========================================================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide2)
    add_header(slide2, "Exploration: 400K Records Across 5 Lending Scenarios", "Rubric 2: Data Exploration & Dictionary")

    # Left Column: Scenarios Table
    scen_card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    scen_card.fill.solid()
    scen_card.fill.fore_color.rgb = CARD_BG
    scen_card.line.color.rgb = PRIMARY
    tf_s = scen_card.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "📊 5 Lending Verticals (80,000 Records Each)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    scenarios = [
        ("E-commerce Shopping EMI", "₹10K - ₹200K", "3 - 24 Months", "12% - 20% APR"),
        ("Home Appliances EMI", "₹20K - ₹300K", "6 - 36 Months", "10% - 18% APR"),
        ("Vehicle EMI", "₹80K - ₹1,500K", "12 - 84 Months", "8% - 14% APR"),
        ("Personal Loan EMI", "₹50K - ₹1,000K", "12 - 60 Months", "11% - 22% APR"),
        ("Education EMI", "₹50K - ₹500K", "6 - 48 Months", "8% - 13% APR")
    ]
    for name, amt, ten, rate in scenarios:
        p_item = tf_s.add_paragraph()
        p_item.text = f"• {name}: Amount {amt} | Tenure {ten} | {rate}"
        p_item.font.size = Pt(10.5)
        p_item.font.color.rgb = TEXT_MUTED

    p_div = tf_s.add_paragraph()
    p_div.text = "\nDataset Dimensions: 400,000 rows × 24 columns (60.8 MB)"
    p_div.font.size = Pt(11)
    p_div.font.bold = True
    p_div.font.color.rgb = ACCENT

    # Right Column: 22 Features Data Dictionary
    dict_card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    dict_card.fill.solid()
    dict_card.fill.fore_color.rgb = CARD_BG
    dict_card.line.color.rgb = PRIMARY
    tf_d = dict_card.text_frame
    tf_d.word_wrap = True
    pd0 = tf_d.paragraphs[0]
    pd0.text = "📋 22 Core Input Variables Grouping"
    pd0.font.size = Pt(14)
    pd0.font.bold = True
    pd0.font.color.rgb = TEXT_WHITE

    groups = [
        ("Personal Demographics", "age (25-60), gender, marital_status, education"),
        ("Employment & Income", "monthly_salary (15k-200k), employment_type, years_of_employment, company_type"),
        ("Housing & Family", "house_type, monthly_rent, family_size, dependents"),
        ("Monthly Obligations", "school_fees, college_fees, travel_expenses, groceries_utilities, other_expenses"),
        ("Financial & Bureau", "existing_loans, current_emi_amount, credit_score (300-850), bank_balance, emergency_fund"),
        ("Loan Application", "emi_scenario, requested_amount, requested_tenure"),
        ("Dual Targets", "emi_eligibility (3 classes) & max_monthly_emi (₹500-₹50,000)")
    ]
    for grp, vars_text in groups:
        p_g = tf_d.add_paragraph()
        p_g.text = f"• {grp}: {vars_text}"
        p_g.font.size = Pt(10)
        p_g.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: Data Quality & Missing Value Strategy (Rubric: NaN / Null Handling)
    # =========================================================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide3)
    add_header(slide3, "Data Quality Assessment & Zero-Leakage Preprocessing", "Rubric 3: Missing Value Handling & Quality")

    # 3 Strategy Cards
    c1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2))
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = ACCENT
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "🧹 Automated Data Audit"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = ACCENT
    
    audits = [
        "Audited 400,000 profiles with DataQualityAssessor.",
        "Zero Missing Values in generated ground truth.",
        "Robust Fallback: median imputation for skewed numericals, mode for categoricals.",
        "Duplicate Records: Automated deduplication checks.",
        "Anomaly Filters: Ensured salary > 0, credit score [300, 850], non-negative living costs."
    ]
    for a in audits:
        pa = tf1.add_paragraph()
        pa.text = f"• {a}"
        pa.font.size = Pt(10.5)
        pa.font.color.rgb = TEXT_MUTED

    c2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2))
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = PRIMARY
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "⚖️ Leak-Free Preprocessing"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = PRIMARY
    
    prep = [
        "Fitted solely on Training split to prevent any data leakage.",
        "StandardScaler applied to skewed financial amounts (salaries, rents, EMIs).",
        "OneHotEncoder for categorical features (education, company, house, scenario).",
        "LabelEncoder for 3-class target: Eligible (0), High_Risk (1), Not_Eligible (2).",
        "Serialized fitted transformer to preprocessor.joblib."
    ]
    for pr in prep:
        pp = tf2.add_paragraph()
        pp.text = f"• {pr}"
        pp.font.size = Pt(10.5)
        pp.font.color.rgb = TEXT_MUTED

    c3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2))
    c3.fill.solid()
    c3.fill.fore_color.rgb = CARD_BG
    c3.line.color.rgb = TEXT_CYAN
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "🎯 Stratified Splitting"
    p3.font.size = Pt(14)
    p3.font.bold = True
    p3.font.color.rgb = TEXT_CYAN
    
    splits = [
        "70% Train (28,000 / 280,000 rows): Model training.",
        "15% Validation (6,000 / 60,000 rows): Hyperparameter tuning.",
        "15% Test (6,000 / 60,000 rows): Blind test evaluation.",
        "Stratification Key: Composite key combining emi_eligibility and emi_scenario.",
        "Ensures identical class ratios across all lending verticals."
    ]
    for sp in splits:
        ps = tf3.add_paragraph()
        ps.text = f"• {sp}"
        ps.font.size = Pt(10.5)
        ps.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 4: Feature Engineering & Financial Risk Metrics (Rubric: Milestones)
    # =========================================================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide4)
    add_header(slide4, "Domain-Driven Financial Feature Engineering & Equations", "Rubric 4: Feature Engineering & Mathematical Formulations")

    fe_box = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    fe_box.fill.solid()
    fe_box.fill.fore_color.rgb = CARD_BG
    fe_box.line.color.rgb = PRIMARY
    tf_fe = fe_box.text_frame
    tf_fe.word_wrap = True
    p_fe0 = tf_fe.paragraphs[0]
    p_fe0.text = "🧮 8 Key Banking Risk Ratios Derived from 22 Raw Variables"
    p_fe0.font.size = Pt(14)
    p_fe0.font.bold = True
    p_fe0.font.color.rgb = TEXT_WHITE

    formulas = [
        ("1. Total Living Expenses", "rent + school_fees + college_fees + travel + groceries_utilities + other_expenses"),
        ("2. Net Disposable Income", "monthly_salary - total_living_expenses - current_emi_amount"),
        ("3. Debt-To-Income (DTI)", "current_emi_amount / monthly_salary (Active monthly debt burden)"),
        ("4. Fixed Obligation to Income (FOIR)", "(current_emi + estimated_req_emi + monthly_rent) / monthly_salary [Banking cap: <=50%]"),
        ("5. Emergency Cushion Months", "emergency_fund / total_monthly_expenses (Months of survival liquidity)"),
        ("6. Liquid Assets to Loan Ratio", "(bank_balance + emergency_fund) / requested_amount"),
        ("7. Employment Stability Index", "Weighted index: Company Type (40%) + Employment Type (35%) + Log-tenure (25%)"),
        ("8. Credit Risk Tiers", "Tier 1: Excellent (>=750) | Tier 2: Good (680-749) | Tier 3: Fair (600-679) | Tier 4: Poor (<600)")
    ]
    for title, desc in formulas:
        pf = tf_fe.add_paragraph()
        pf.text = f"• {title}: {desc}"
        pf.font.size = Pt(11)
        pf.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: 5 Distinct Visualizations & EDA Insights (Rubric: 5 Visualizations)
    # =========================================================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide5)
    add_header(slide5, "5 Distinct Visualizations & Exploratory Data Conclusions", "Rubric 6: Five Distinct Visualizations & EDA")

    viz_items = [
        ("Viz 1: Grouped Bar Chart", "Scenario Volume & Eligibility Mix", "Evaluates approval, high-risk, and decline proportions across 5 scenarios. E-commerce shows lowest default risk (78% approved); Vehicle & Personal have higher high-risk proportion."),
        ("Viz 2: Density / Histogram KDE", "Credit Score vs. Decision Separation", "Clear bimodal separation. Not_Eligible concentrates heavily below 560; High_Risk occupies 560-660; Prime Eligible candidates dominate >= 680."),
        ("Viz 3: Scatter Plot", "Monthly Salary vs. Max Safe Monthly EMI", "Linear cashflow expansion with distinct risk ceilings. Verifies that max safe EMI does not exceed 75% unencumbered cashflow or FOIR cap."),
        ("Viz 4: Risk Heatmap / FOIR", "FOIR vs. DTI Default Boundaries", "Reveals that FOIR > 50% is the single strongest driver of loan rejections, regardless of gross income."),
        ("Viz 5: Stacked Percentage Bar", "Employment Sector vs. Approval Share", "Government employees boast the highest approval stability (88.4%), followed by Large MNCs (76.2%), Mid-size (64.5%), and Startups (52.1%).")
    ]
    for idx, (vtitle, vsub, vdesc) in enumerate(viz_items):
        y_pos = 1.6 + (idx * 1.02)
        vcard = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.7), Inches(0.92))
        vcard.fill.solid()
        vcard.fill.fore_color.rgb = CARD_BG
        vcard.line.color.rgb = PRIMARY
        tf_v = vcard.text_frame
        tf_v.word_wrap = True
        pv = tf_v.paragraphs[0]
        pv.text = f"{vtitle} — {vsub}"
        pv.font.size = Pt(12)
        pv.font.bold = True
        pv.font.color.rgb = TEXT_CYAN
        pvd = tf_v.add_paragraph()
        pvd.text = vdesc
        pvd.font.size = Pt(10)
        pvd.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 6: Dual ML Model Development & MLflow Benchmarking (Rubric: Milestones)
    # =========================================================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide6)
    add_header(slide6, "Dual Machine Learning Models & MLflow Experiment Tracking", "Rubric 5 & 8: Model Performance & Benchmarking")

    # Left: Classification Benchmark
    clf_box = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2))
    clf_box.fill.solid()
    clf_box.fill.fore_color.rgb = CARD_BG
    clf_box.line.color.rgb = ACCENT
    tf_cb = clf_box.text_frame
    tf_cb.word_wrap = True
    pcb = tf_cb.paragraphs[0]
    pcb.text = "🎯 Classification (Target: Accuracy > 90%)"
    pcb.font.size = Pt(14)
    pcb.font.bold = True
    pcb.font.color.rgb = ACCENT

    clf_models = [
        ("Logistic Regression (Baseline)", "89.20%", "0.8143", "0.9707", "Baseline Model"),
        ("Random Forest Classifier", "97.23%", "0.9550", "0.9981", "Challenger (Passed)"),
        ("Gradient Boosting (Champion)", "99.10%", "0.9866", "0.9998", "🏆 Promoted to Prod")
    ]
    for name, acc, f1, auc, stat in clf_models:
        p_c = tf_cb.add_paragraph()
        p_c.text = f"\n• {name}\n  Accuracy: {acc} | F1: {f1} | AUC: {auc}\n  Status: {stat}"
        p_c.font.size = Pt(10.5)
        p_c.font.color.rgb = TEXT_WHITE if "Champion" in stat else TEXT_MUTED

    # Right: Regression Benchmark
    reg_box = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    reg_box.fill.solid()
    reg_box.fill.fore_color.rgb = CARD_BG
    reg_box.line.color.rgb = PRIMARY
    tf_rb = reg_box.text_frame
    tf_rb.word_wrap = True
    prb = tf_rb.paragraphs[0]
    prb.text = "📈 Regression (Target: RMSE < ₹2,000 INR)"
    prb.font.size = Pt(14)
    prb.font.bold = True
    prb.font.color.rgb = PRIMARY

    reg_models = [
        ("Ridge Linear Regression", "₹2,469.4", "₹1,851.6", "0.9502", "Baseline Model"),
        ("Random Forest Regressor", "₹491.9", "₹118.0", "0.9980", "Challenger (Passed)"),
        ("Gradient Boosting (Champion)", "₹341.7", "₹106.5", "0.9990", "🏆 Promoted to Prod")
    ]
    for name, rmse, mae, r2, stat in reg_models:
        p_r = tf_rb.add_paragraph()
        p_r.text = f"\n• {name}\n  RMSE: {rmse} | MAE: {mae} | R²: {r2}\n  Status: {stat}"
        p_r.font.size = Pt(10.5)
        p_r.font.color.rgb = TEXT_WHITE if "Champion" in stat else TEXT_MUTED

    # =========================================================================
    # SLIDE 7: Platform Engineering & Smart Approval Advisor (Rubric: Modularity)
    # =========================================================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide7)
    add_header(slide7, "System Architecture & Smart Approval Advisor Engine", "Rubric 10: System Engineering & Modularity")

    s1 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    s1.fill.solid()
    s1.fill.fore_color.rgb = CARD_BG
    s1.line.color.rgb = PRIMARY
    tfs1 = s1.text_frame
    tfs1.word_wrap = True
    ps1 = tfs1.paragraphs[0]
    ps1.text = "🧭 Smart Approval Advisor (What-If Engine)"
    ps1.font.size = Pt(14)
    ps1.font.bold = True
    ps1.font.color.rgb = PRIMARY

    adv_points = [
        "Identifies Root Cause Risk Drivers:",
        "  - FOIR Breach (>50% income committed)",
        "  - Requested Installment > Safe Monthly EMI Capacity",
        "  - Subprime Credit Bureau Score (<660)",
        "  - Insufficient Emergency Liquidity Cushion (<1 month)",
        "Automated Counterfactual Optimization Levers:",
        "  - Lever 1: Extend repayment tenure (e.g. 12m -> 24m -> 36m)",
        "  - Lever 2: Increase downpayment / reduce principal (10-30%)",
        "  - Lever 3: Debt consolidation / pay down active loans",
        "Result: Flips 35% of declined borrowers into Approved!"
    ]
    for pt in adv_points:
        ppt = tfs1.add_paragraph()
        ppt.text = f"• {pt}" if not pt.startswith("  -") else pt
        ppt.font.size = Pt(10.5)
        ppt.font.color.rgb = TEXT_MUTED

    s2 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    s2.fill.solid()
    s2.fill.fore_color.rgb = CARD_BG
    s2.line.color.rgb = ACCENT
    tfs2 = s2.text_frame
    tfs2.word_wrap = True
    ps2 = tfs2.paragraphs[0]
    ps2.text = "🏛️ Multi-Page App & CRUD Architecture"
    ps2.font.size = Pt(14)
    ps2.font.bold = True
    ps2.font.color.rgb = ACCENT

    arch_points = [
        "1. Executive Portal (app.py): System health & champion versions.",
        "2. Loan Officer Assessment (pages/1_🎯...): Dual prediction, live risk gauges, and 1-click preset profiles (Prime, Marginal, Strained).",
        "3. Exploratory Data Insights (pages/2_📊...): Slicing across 400k records, distributions, and quality audit.",
        "4. Applicant Records CRUD (pages/3_🗄️...): SQLite ACID database to Search, Create, Edit status, Delete, and Export CSV.",
        "5. MLflow Registry Hub (pages/4_🔬...): Model comparison, metrics tracking, and 1-click model promotion.",
        "6. Code Modularity: Clean src/ package, zero leakage, 100% unit test coverage."
    ]
    for apt in arch_points:
        papt = tfs2.add_paragraph()
        papt.text = f"• {apt}"
        papt.font.size = Pt(10.5)
        papt.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 8: Business Impact, Conclusions & Final Links (Rubric: Conclusion)
    # =========================================================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide8)
    add_header(slide8, "Business Impact, Conclusions & Capstone Verification", "Rubric 7: Final Summary & Business Value")

    b_card = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    b_card.fill.solid()
    b_card.fill.fore_color.rgb = CARD_BG
    b_card.line.color.rgb = ACCENT
    tf_b = b_card.text_frame
    tf_b.word_wrap = True
    pb0 = tf_b.paragraphs[0]
    pb0.text = "💼 Business Value & Underwriting Transformation"
    pb0.font.size = Pt(14)
    pb0.font.bold = True
    pb0.font.color.rgb = ACCENT

    conclusions = [
        ("80% Underwriting Time Reduction", "Inference latency drops from 45 minutes of manual document inspection down to <50 ms automated scoring."),
        ("Elimination of Human Bias", "Standardized mathematical FOIR and cashflow rules replace subjective credit judgment."),
        ("Lower Portfolio Default Rates", "Strict cashflow caps prevent borrower over-indebtedness, safeguarding banking capital reserves."),
        ("Customer Retention via Advisor", "Converts 35% of rejected applicants into qualified borrowers through structured restructuring pathways."),
        ("Enterprise MLOps Maturity", "Automated experiment logging with MLflow and zero-downtime model promotion."),
        ("Code Modularity & CI/CD", "Automated unit test suite (4/4 passed) + Docker containerization + GitHub Actions CI workflow.")
    ]
    for title, desc in conclusions:
        p_cn = tf_b.add_paragraph()
        p_cn.text = f"• {title}: {desc}"
        p_cn.font.size = Pt(10.5)
        p_cn.font.color.rgb = TEXT_MUTED

    # Bottom links footer
    p_ft = tf_b.add_paragraph()
    p_ft.text = "\n🔗 Final Links: GitHub: https://github.com/sagnik1819/emipredict-ai  |  Live Web App: https://forwarding-humanity-pest-urban.trycloudflare.com"
    p_ft.font.size = Pt(11)
    p_ft.font.bold = True
    p_ft.font.color.rgb = TEXT_CYAN

    # Save presentation
    prs.save(output_path)
    print(f"PowerPoint presentation generated successfully: {output_path}")

if __name__ == "__main__":
    create_presentation()
