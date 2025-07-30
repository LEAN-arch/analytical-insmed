# ======================================================================================
# ANALYTICAL DEVELOPMENT OPERATIONS COMMAND CENTER
#
# A single-file Streamlit application for the Associate Director, AD Operations.
#
# VERSION: Scientific & Regulatory Compliance Edition (Full SPC & Regulations)
#
# This dashboard provides a real-time, strategic, and scientifically-grounded
# view of the Analytical Development Operations function. It is designed to manage a
# high-throughput testing team, optimize analytical methods using advanced statistics,
# oversee the technology transfer lifecycle, and support the broader process
# development pipeline for biologic drug candidates (e.g., AAVs).
#
# It integrates principles from:
#   - ICH Q2(R1) Validation of Analytical Procedures
#   - ICH Q8(R2) Pharmaceutical Development
#   - ICH Q14 Analytical Procedure Development
#   - cGMP, FDA, & EMA guidelines
#   - ALCOA+ Data Integrity Principles
#
# To Run:
# 1. Save this code as 'ad_ops_dashboard_final.py'
# 2. Create 'requirements.txt' with specified libraries.
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run from your terminal: streamlit run ad_ops_dashboard_final.py
#
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================================================================================
# SECTION 1: APP CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(
    page_title="AD Ops Command Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main .block-container { padding: 1rem 3rem 3rem; }
    .stMetric { background-color: #fcfcfc; border: 1px solid #e0e0e0; border-left: 5px solid #673ab7; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; box-shadow: 0 -2px 5px rgba(0,0,0,0.1); border-bottom-color: #FFFFFF !important; }
    .st-expander { border: 1px solid #E0E0E0 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# SECTION 2: SME-DRIVEN DATA SIMULATION FOR ANALYTICAL DEVELOPMENT
# ======================================================================================
@st.cache_data(ttl=600)
def generate_master_data():
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=52, freq='W-MON')); sample_data = {'Week': dates, 'Samples_Received': np.random.randint(80, 150, 52), 'Samples_Tested': np.random.randint(70, 140, 52)}; sample_df = pd.DataFrame(sample_data); sample_df['Backlog'] = sample_df['Samples_Received'].cumsum() - sample_df['Samples_Tested'].cumsum()
    control_dates = pd.to_datetime(pd.date_range(start='2023-06-01', periods=100, freq='D')); cqa_data = {'Date': control_dates, 'AAV_Titer_Control': np.random.normal(1.0e13, 0.05e13, 100)}; cqa_data['AAV_Titer_Control'][80:] += 0.15e13; cqa_df = pd.DataFrame(cqa_data)
    methods = ['AAV Titer (ddPCR)', 'Capsid Purity (HPLC-SEC)', 'Host Cell DNA (qPCR)', 'Potency (Cell-Based Assay)', 'Endotoxin (LAL)']; transfer_data = {'Method': methods * 2, 'Program': ['AAV-101']*5 + ['AAV-201']*5, 'Receiving_Unit': ['Internal QC', 'CDMO-A', 'Internal QC', 'CDMO-B', 'Internal QC'] * 2, 'Status': np.random.choice(['Development', 'Optimization', 'Validation', 'Transferred', 'Failed'], 10, p=[0.1,0.2,0.4,0.2,0.1]), 'Complexity_Score': np.random.randint(3, 10, 10), 'SOP_Maturity_Score': np.random.randint(4, 10, 10), 'Training_Cycles': np.random.randint(1, 4, 10)}; transfer_df = pd.DataFrame(transfer_data); transfer_df['Transfer_Success'] = ((transfer_df['Status'] == 'Transferred').astype(int) + (transfer_df['Complexity_Score'] < 5) + (transfer_df['SOP_Maturity_Score'] > 7)) > 1
    X = np.random.uniform(-1, 1, (15, 2)); doe_df = pd.DataFrame(X, columns=['pH', 'Gradient_Slope_Pct_min']); doe_df['AAV_Purity_Pct'] = 95 - 2*doe_df['pH']**2 - 3*doe_df['Gradient_Slope_Pct_min']**2 + doe_df['pH']*doe_df['Gradient_Slope_Pct_min'] + np.random.normal(0, 0.5, 15)
    team_data = {'Scientist': ['J. Doe', 'S. Smith', 'M. Lee', 'K. Chen'], 'Role': ['Sr. Scientist', 'Scientist II', 'Scientist I', 'RA II'], 'Expertise': ['HPLC', 'ddPCR', 'ELISA', 'CE']}; team_df = pd.DataFrame(team_data); equipment_data = {'Instrument': ['HPLC-01', 'HPLC-02', 'CE-01', 'ddPCR-01'], 'Status': ['Online', 'Online', 'Calibration Due', 'Offline - Maintenance']}; equipment_df = pd.DataFrame(equipment_data)
    subgroup_size = 5; num_subgroups = 40; subgroup_data = []
    for i in range(num_subgroups):
        mean = 100; std_dev = 5
        if i >= 20: std_dev = 10
        if i >= 30: mean = 110
        replicates = np.random.normal(mean, std_dev, subgroup_size)
        for j, rep in enumerate(replicates): subgroup_data.append({'Subgroup_ID': i + 1, 'Replicate': j + 1, 'Potency_Result': rep})
    subgroup_df = pd.DataFrame(subgroup_data)
    p_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=24, freq='ME')); batches_tested = np.random.randint(15, 25, 24); base_fail_rate = 0.05; batches_failed = np.random.binomial(n=batches_tested, p=base_fail_rate); batches_failed[18] = 5; p_chart_data = {'Month': p_dates, 'Batches_Tested': batches_tested, 'Batches_Failed': batches_failed}; p_chart_df = pd.DataFrame(p_chart_data)
    return sample_df, cqa_df, transfer_df, doe_df, team_df, equipment_df, subgroup_df, p_chart_df

# ======================================================================================
# SECTION 3: ADVANCED ANALYTICAL & ML MODELS
# ======================================================================================
@st.cache_resource
def get_transfer_risk_model(df):
    features = ['Complexity_Score', 'SOP_Maturity_Score', 'Training_Cycles']; target = 'Transfer_Success'
    X, y = df[features], df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'); model.fit(X, y)
    return model

def plot_rsm_surface(df, x_col, y_col, z_col):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    x = np.linspace(df[x_col].min(), df[x_col].max(), 30); y = np.linspace(df[y_col].min(), df[y_col].max(), 30)
    x_grid, y_grid = np.meshgrid(x, y)
    poly = PolynomialFeatures(degree=2); X_poly = poly.fit_transform(df[[x_col, y_col]]); model = LinearRegression(); model.fit(X_poly, df[z_col])
    X_pred_poly = poly.transform(np.c_[x_grid.ravel(), y_grid.ravel()]); z_grid = model.predict(X_pred_poly).reshape(x_grid.shape)
    fig = go.Figure(data=[go.Surface(z=z_grid, x=x, y=y, colorscale='Viridis', name='Response Surface', showlegend=True)])
    fig.add_trace(go.Scatter3d(x=df[x_col], y=df[y_col], z=df[z_col], mode='markers', marker=dict(size=5, color='red', symbol='circle'), name='DOE Experimental Points'))
    fig.update_layout(title='<b>Response Surface: Method Optimization</b>', scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col), margin=dict(l=0, r=0, b=0, t=40), legend=dict(x=0.8, y=0.9))
    return fig

def plot_i_mr_chart(df, value_col):
    individuals = df[value_col]; moving_ranges = abs(individuals.diff()).dropna()
    i_mean = individuals.mean(); i_ucl = i_mean + 3 * moving_ranges.mean() / 1.128; i_lcl = i_mean - 3 * moving_ranges.mean() / 1.128
    fig = go.Figure(); fig.add_trace(go.Scatter(y=individuals, name='Individual Value', mode='lines+markers', line=dict(color='#673ab7'))); fig.add_hline(y=i_mean, line=dict(color='green', dash='dot'), name='Mean'); fig.add_hline(y=i_ucl, line=dict(color='red', dash='dash'), name='UCL'); fig.add_hline(y=i_lcl, line=dict(color='red', dash='dash'), name='LCL')
    outliers = individuals[(individuals > i_ucl) | (individuals < i_lcl)]; fig.add_trace(go.Scatter(x=outliers.index, y=outliers, mode='markers', name='Out of Control Signal', marker=dict(symbol='x', color='red', size=12)))
    fig.update_layout(title=f'<b>I-Chart: Method Performance Monitoring for {value_col}</b>', yaxis_title='Value', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_xbar_r_chart(df):
    subgroup_size = df['Replicate'].max()
    stats = df.groupby('Subgroup_ID')['Potency_Result'].agg(['mean', 'max', 'min']).reset_index()
    stats['range'] = stats['max'] - stats['min']
    A2, D3, D4 = 0.577, 0, 2.114 # Shewhart constants for n=5
    x_bar_bar = stats['mean'].mean(); r_bar = stats['range'].mean()
    x_ucl = x_bar_bar + A2 * r_bar; x_lcl = x_bar_bar - A2 * r_bar
    r_ucl = r_bar * D4; r_lcl = r_bar * D3
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart (Process Mean)", "R-Chart (Process Variation)"))
    fig.add_trace(go.Scatter(x=stats['Subgroup_ID'], y=stats['mean'], name='Subgroup Mean', mode='lines+markers'), row=1, col=1)
    fig.add_hline(y=x_bar_bar, line=dict(color='green', dash='dot'), name='Center Line', row=1, col=1)
    fig.add_hline(y=x_ucl, line=dict(color='red', dash='dash'), name='UCL', row=1, col=1); fig.add_hline(y=x_lcl, line=dict(color='red', dash='dash'), name='LCL', row=1, col=1)
    fig.add_trace(go.Scatter(x=stats['Subgroup_ID'], y=stats['range'], name='Subgroup Range', mode='lines+markers', line_color='orange'), row=2, col=1)
    fig.add_hline(y=r_bar, line=dict(color='green', dash='dot'), name='Center Line', row=2, col=1)
    fig.add_hline(y=r_ucl, line=dict(color='red', dash='dash'), name='UCL', row=2, col=1); fig.add_hline(y=r_lcl, line=dict(color='red', dash='dash'), name='LCL', row=2, col=1)
    fig.update_layout(height=600, title_text="<b>X-bar & R Chart for Subgroup Data (Potency Assay)</b>")
    return fig

def plot_p_chart(df):
    df['proportion'] = df['Batches_Failed'] / df['Batches_Tested']
    p_bar = df['Batches_Failed'].sum() / df['Batches_Tested'].sum()
    df['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df['Batches_Tested'])
    df['LCL'] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df['Batches_Tested'])).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Month'], y=df['proportion'], name='Proportion Failed', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df['Month'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Month'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color='red', dash='dash')))
    fig.add_hline(y=p_bar, name='Average Fail Rate', line=dict(color='green', dash='dot'))
    outliers = df[df['proportion'] > df['UCL']]; fig.add_trace(go.Scatter(x=outliers['Month'], y=outliers['proportion'], mode='markers', name='Out of Control Signal', marker=dict(symbol='x', color='red', size=12)))
    fig.update_layout(title='<b>p-Chart for Batch Release Failure Rate</b>', xaxis_title='Month', yaxis_title='Proportion of Batches Failed', yaxis_tickformat=".2%")
    return fig

# ======================================================================================
# SECTION 4: MAIN APPLICATION LAYOUT & SCIENTIFIC NARRATIVE
# ======================================================================================
st.title("🧬 Analytical Development Operations Command Center")
st.markdown("##### A strategic dashboard for managing high-throughput testing, method lifecycle, and program leadership in biologics development.")
sample_df, cqa_df, transfer_df, doe_df, team_df, equipment_df, subgroup_df, p_chart_df = generate_master_data()
transfer_risk_model = get_transfer_risk_model(transfer_df)

st.markdown("### I. AD Operations & Sample Throughput Command Center")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
avg_tat = (sample_df['Backlog'].mean() / sample_df['Samples_Tested'].mean()) * 7 if sample_df['Samples_Tested'].mean() > 0 else 0; kpi_col1.metric("Avg. Sample TAT (Days)", f"{avg_tat:.1f}")
kpi_col2.metric("Weekly Throughput", f"{sample_df['Samples_Tested'].iloc[-1]} Samples", f"{sample_df['Samples_Tested'].iloc[-1] - sample_df['Samples_Tested'].iloc[-2]:+d} vs last week")
kpi_col3.metric("Methods in Transfer", f"{transfer_df[transfer_df['Status'].isin(['Validation', 'Optimization'])].shape[0]}")
equipment_offline = equipment_df[equipment_df['Status'] != 'Online'].shape[0]; kpi_col4.metric("Equipment Readiness", f"{100-((equipment_offline/len(equipment_df))*100):.0f}%", f"{equipment_offline} Instrument(s) Offline", "inverse")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["**II. METHOD PERFORMANCE & OPTIMIZATION (DOE/SPC)**", "**III. TECHNOLOGY TRANSFER & VALIDATION (ML)**", "**IV. PROGRAM LEADERSHIP & RESOURCES**", "**V. HIGH THROUGHPUT TECHNOLOGY**"])

with tab1:
    st.header("II. Method Performance & Optimization")
    st.markdown("_This section provides tools for developing robust analytical procedures using advanced statistical methods and for monitoring their performance over the lifecycle, in accordance with ICH Q8, Q14, and Q2._")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. Method Optimization via Response Surface Methodology (RSM)")
        with st.expander("View Methodological Summary", expanded=False): st.markdown("""...""")
        st.plotly_chart(plot_rsm_surface(doe_df, 'pH', 'Gradient_Slope_Pct_min', 'AAV_Purity_Pct'), use_container_width=True)
    with col2:
        st.subheader("B. Method Lifecycle Monitoring via Statistical Process Control (SPC) Suite")
        spc_choice = st.selectbox("Select SPC Analysis Type:", ["Method Stability (I-MR Chart)", "Subgroup Precision & Accuracy (X-bar & R Chart)", "Process Yield (p-Chart)"])
        if spc_choice == "Method Stability (I-MR Chart)":
            with st.expander("View Methodological Summary", expanded=True): st.markdown("""...""")
            st.plotly_chart(plot_i_mr_chart(cqa_df, 'AAV_Titer_Control'), use_container_width=True)
        elif spc_choice == "Subgroup Precision & Accuracy (X-bar & R Chart)":
            with st.expander("View Methodological Summary", expanded=True): st.markdown("""...""")
            st.plotly_chart(plot_xbar_r_chart(subgroup_df), use_container_width=True)
        elif spc_choice == "Process Yield (p-Chart)":
            with st.expander("View Methodological Summary", expanded=True): st.markdown("""...""")
            st.plotly_chart(plot_p_chart(p_chart_df), use_container_width=True)

with tab2:
    st.header("III. Technology Transfer & Validation")
    st.markdown("_This section provides tools to manage the transfer of analytical procedures to receiving units (QC labs, CDMOs) and to predict the likelihood of success based on key attributes._")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("A. Tech Transfer Risk Prediction (ML)")
        with st.expander("View Methodological Summary", expanded=False): st.markdown("""...""")
        complexity = st.slider("Method Complexity (1-10)", 1, 10, 5)
        sop_maturity = st.slider("SOP Maturity (1-10)", 1, 10, 7)
        training_cycles = st.slider("Planned Training Cycles", 1, 5, 2)
        if st.button("🔬 Predict Transfer Success", type="primary"):
            input_df = pd.DataFrame([[complexity, sop_maturity, training_cycles]], columns=['Complexity_Score', 'SOP_Maturity_Score', 'Training_Cycles'])
            prediction_proba = transfer_risk_model.predict_proba(input_df)[0][1]
            st.success(f"Predicted Probability of Successful Transfer: **{prediction_proba:.1%}**")
    with col2:
        st.subheader("B. Method Transfer Portfolio Status")
        st.dataframe(transfer_df[['Program', 'Method', 'Receiving_Unit', 'Status']], use_container_width=True)

with tab3:
    st.header("IV. Program Leadership & Resource Management")
    st.markdown("_This section provides oversight of the AD Ops team's alignment with high-level drug development programs and the status of critical laboratory resources._")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. Sample Testing Capacity vs. Demand")
        fig = go.Figure(); fig.add_trace(go.Bar(x=sample_df['Week'], y=sample_df['Samples_Received'], name='Samples Received', marker_color='lightgrey')); fig.add_trace(go.Scatter(x=sample_df['Week'], y=sample_df['Backlog'], name='Sample Backlog', yaxis='y2', line=dict(color='red'))); fig.update_layout(title='<b>Weekly Sample Load & Backlog Trend</b>', yaxis_title='Weekly Sample Count', yaxis2=dict(title='Total Backlog', overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("B. Critical Equipment Status")
        for _, row in equipment_df.iterrows():
            if row['Status'] == 'Online': st.success(f"**{row['Instrument']}:** {row['Status']}")
            elif row['Status'] == 'Calibration Due': st.warning(f"**{row['Instrument']}:** {row['Status']}")
            else: st.error(f"**{row['Instrument']}:** {row['Status']}")
    
with tab4:
    st.header("V. High Throughput Technology Assessment")
    st.markdown("_This section focuses on identifying and prioritizing opportunities for automation and high-throughput solutions to accelerate process development._")
    tech_data = {'Technology': ['Robotic Liquid Handler', 'Automated Plate Reader', 'High-Throughput HPLC', 'Microfluidics Platform'], 'Targeted_Process': ['Sample Preparation', 'ELISA/Potency Assays', 'Purity/Impurity Testing', 'Early-Stage Screening'], 'Est_Throughput_Increase': [5, 3, 4, 10], 'Est_FTE_Saving': [1.5, 0.5, 1.0, 0.75], 'Implementation_Complexity': ['High', 'Low', 'Medium', 'High'], 'Project_Status': ['Evaluation', 'Budgeting', 'Not Started', 'Feasibility']}
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True)

# ============================ SIDEBAR ============================
st.sidebar.image("https://assets-global.website-files.com/62a269e3ea783635a1608298/62a269e3ea783626786083d9_logo-horizontal.svg", use_container_width=True)
st.sidebar.markdown("### Role Focus")
st.sidebar.info("This dashboard is for an **Associate Director, AD Operations**, focused on building a high-throughput testing function, leading method development & transfer, and acting as a PDT representative for biologics.")

# --- ENHANCEMENT: New section for all applicable regulations ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Applicable Regulatory Frameworks")
with st.sidebar.expander("View Key GxP and ISO Regulations", expanded=False):
    st.markdown("""
    **ICH Q2(R1) - Validation of Analytical Procedures**
    - The core guideline defining validation characteristics.
    - *"...specificity, linearity, range, accuracy, precision (repeatability, intermediate precision), detection limit, quantitation limit, robustness."*

    **ICH Q8(R2) - Pharmaceutical Development**
    - Introduces Quality by Design (QbD) concepts.
    - *"The aim of pharmaceutical development is to design a quality product and its manufacturing process to consistently deliver the intended performance..."*
    
    **ICH Q9 - Quality Risk Management**
    - The framework for risk-based decision-making.
    - *"The evaluation of the risk to quality should be based on scientific knowledge and ultimately link to the protection of the patient."*

    **ICH Q14 - Analytical Procedure Development**
    - Formalizes the lifecycle and QbD approach for methods.
    - *"This guideline describes science and risk-based approaches for developing and maintaining analytical procedures suitable for the assessment of the quality of drug substances and drug products."*

    **21 CFR Part 211 - cGMP for Finished Pharmaceuticals**
    - US FDA's enforceable regulations for manufacturing and testing.
    - *§211.165(e): "The accuracy, sensitivity, specificity, and reproducibility of test methods employed by the firm shall be established and documented."*
    
    **21 CFR Part 11 - Electronic Records; Electronic Signatures**
    - Governs data integrity for all computerized lab systems (LIMS, ELN, CDS).
    - *"Applicability to records in electronic form that are created, modified, maintained, archived, retrieved, or transmitted, under any records requirements..."*

    **EudraLex Vol. 4, Ch 6 - Quality Control**
    - The EMA's GMP requirements for QC laboratories.
    - *"Test methods should be validated... before they are brought into routine use."*

    **ISO 17025:2017 - General requirements for the competence of testing and calibration laboratories**
    - The international standard for laboratory quality management.
    - *"The laboratory shall be responsible for the impartiality of its laboratory activities and shall not allow... pressures to compromise impartiality."*
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### Key Scientific Concepts")
st.sidebar.markdown("""
- **DOE/RSM:** Design of Experiments / Response Surface Methodology for efficient, multi-factorial process optimization.
- **SPC:** Statistical Process Control for monitoring the state of control of a validated process over its lifecycle.
- **AAV:** Adeno-Associated Virus, a common vector for gene therapy products.
- **ALCOA+:** Attributable, Legible, Contemporaneous, Original, Accurate (+ Complete, Consistent, Enduring, Available). Principles of data integrity.
""")
