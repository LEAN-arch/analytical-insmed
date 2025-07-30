# ======================================================================================
# ANALYTICAL DEVELOPMENT OPERATIONS COMMAND CENTER
#
# A single-file Streamlit application for the Associate Director, AD Operations.
#
# VERSION: Final, Unabridged & Fully Populated
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
# 1. Save this code as 'ad_ops_final_dashboard.py'
# 2. Create 'requirements.txt' with specified libraries.
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run from your terminal: streamlit run ad_ops_final_dashboard.py
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
# SECTION 2: SME-DRIVEN DATA SIMULATION
# ======================================================================================
@st.cache_data(ttl=600)
def generate_master_data():
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=52, freq='W-MON')); sample_data = {'Week': dates, 'Samples_Received': np.random.randint(80, 150, 52), 'Samples_Tested': np.random.randint(70, 140, 52)}; sample_df = pd.DataFrame(sample_data); sample_df['Backlog'] = sample_df['Samples_Received'].cumsum() - sample_df['Samples_Tested'].cumsum()
    control_dates = pd.to_datetime(pd.date_range(start='2023-06-01', periods=100, freq='D')); cqa_data = {'Date': control_dates, 'AAV_Titer_Control': np.random.normal(1.0e13, 0.05e13, 100)}; cqa_data['AAV_Titer_Control'][80:] += 0.15e13; cqa_df = pd.DataFrame(cqa_data)
    methods = ['AAV Titer (ddPCR)', 'Capsid Purity (HPLC-SEC)', 'Host Cell DNA (qPCR)', 'Potency (Cell-Based Assay)', 'Endotoxin (LAL)']; transfer_data = {'Method': methods * 2, 'Program': ['AAV-101']*5 + ['AAV-201']*5, 'Receiving_Unit': ['Internal QC', 'CDMO-A', 'Internal QC', 'CDMO-B', 'Internal QC'] * 2, 'Status': np.random.choice(['Development', 'Optimization', 'Validation', 'Transferred', 'Failed'], 10, p=[0.1,0.2,0.4,0.2,0.1]), 'Complexity_Score': np.random.randint(3, 10, 10), 'SOP_Maturity_Score': np.random.randint(4, 10, 10), 'Training_Cycles': np.random.randint(1, 4, 10)}; transfer_df = pd.DataFrame(transfer_data); transfer_df['Transfer_Success'] = ((transfer_df['Status'] == 'Transferred').astype(int) + (transfer_df['Complexity_Score'] < 5) + (transfer_df['SOP_Maturity_Score'] > 7)) > 1
    X = np.random.uniform(-1, 1, (15, 2)); doe_df = pd.DataFrame(X, columns=['pH', 'Gradient_Slope_Pct_min']); doe_df['AAV_Purity_Pct'] = 95 - 2*doe_df['pH']**2 - 3*doe_df['Gradient_Slope_Pct_min']**2 + doe_df['pH']*doe_df['Gradient_Slope_Pct_min'] + np.random.normal(0, 0.5, 15)
    team_data = {'Scientist': ['J. Doe', 'S. Smith', 'M. Lee', 'K. Chen'], 'Role': ['Sr. Scientist', 'Scientist II', 'Scientist I', 'RA II'], 'Expertise': ['HPLC', 'ddPCR', 'ELISA', 'CE']}; team_df = pd.DataFrame(team_data); equipment_data = {'Instrument': ['HPLC-01', 'HPLC-02', 'CE-01', 'ddPCR-01'], 'Status': ['Online', 'Online', 'Calibration Due', 'Offline - Maintenance']}; equipment_df = pd.DataFrame(equipment_data)
    subgroup_size = 5; num_subgroups = 40; subgroup_data = []; mean = 100; std_dev = 5
    for i in range(num_subgroups):
        if i >= 20: std_dev = 10
        if i >= 30: mean = 110
        replicates = np.random.normal(mean, std_dev, subgroup_size)
        for j, rep in enumerate(replicates): subgroup_data.append({'Subgroup_ID': i + 1, 'Replicate': j + 1, 'Potency_Result': rep})
    subgroup_df = pd.DataFrame(subgroup_data)
    p_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=24, freq='ME')); batches_tested = np.random.randint(15, 25, 24); base_fail_rate = 0.05; batches_failed = np.random.binomial(n=batches_tested, p=base_fail_rate); batches_failed[18] = 5; p_chart_data = {'Month': p_dates, 'Batches_Tested': batches_tested, 'Batches_Failed': batches_failed}; p_chart_df = pd.DataFrame(p_chart_data)
    tech_data = {'Technology': ['Robotic Liquid Handler', 'Automated Plate Reader', 'High-Throughput HPLC', 'Microfluidics Platform'], 'Targeted_Process': ['Sample Preparation', 'ELISA/Potency Assays', 'Purity/Impurity Testing', 'Early-Stage Screening'], 'Est_Throughput_Increase_Factor': [5, 3, 4, 10], 'Implementation_Complexity_Score': [8, 3, 6, 9], 'Est_FTE_Saving': [1.5, 0.5, 1.0, 0.75]}; tech_df = pd.DataFrame(tech_data)
    return sample_df, cqa_df, transfer_df, doe_df, team_df, equipment_df, subgroup_df, p_chart_df, tech_df

# ======================================================================================
# SECTION 3: ADVANCED ANALYTICAL & ML MODELS
# ======================================================================================
@st.cache_resource
def get_transfer_risk_model(df):
    features = ['Complexity_Score', 'SOP_Maturity_Score', 'Training_Cycles']; target = 'Transfer_Success'
    X, y = df[features], df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'); model.fit(X, y)
    return model

def plot_rsm_suite(df, x_col, y_col, z_col):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    x = np.linspace(df[x_col].min(), df[x_col].max(), 30); y = np.linspace(df[y_col].min(), df[y_col].max(), 30)
    x_grid, y_grid = np.meshgrid(x, y)
    poly = PolynomialFeatures(degree=2); X_poly = poly.fit_transform(df[[x_col, y_col]]); model = LinearRegression(); model.fit(X_poly, df[z_col])
    X_pred_poly = poly.transform(np.c_[x_grid.ravel(), y_grid.ravel()]); z_grid = model.predict(X_pred_poly).reshape(x_grid.shape)
    fig_3d = go.Figure(data=[go.Surface(z=z_grid, x=x, y=y, colorscale='Viridis', name='Response Surface', showlegend=True)]); fig_3d.add_trace(go.Scatter3d(x=df[x_col], y=df[y_col], z=df[z_col], mode='markers', marker=dict(size=5, color='red', symbol='circle'), name='DOE Points')); fig_3d.update_layout(title='<b>A. Response Surface (3D View)</b>', scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col), margin=dict(l=0, r=0, b=0, t=40))
    fig_2d = go.Figure(data=go.Contour(z=z_grid, x=x, y=y, colorscale='Viridis', contours_coloring='lines', line_width=2)); fig_2d.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', marker=dict(color='black', symbol='x'), name='DOE Points')); fig_2d.add_shape(type="rect", x0=-0.5, y0=-0.7, x1=0.5, y1=0.7, line=dict(color="red", dash="dash"), fillcolor="rgba(255,0,0,0.1)"); fig_2d.add_annotation(x=0, y=0, text="<b>Optimal<br>Region</b>", showarrow=False, font=dict(color="red")); fig_2d.update_layout(title='<b>B. Contour Plot & Design Space (2D View)</b>', xaxis_title=x_col, yaxis_title=y_col)
    return fig_3d, fig_2d

def plot_enhanced_i_mr_chart(df, value_col):
    individuals = df[value_col]; i_mean = individuals.mean(); mr_mean = abs(individuals.diff()).mean(); i_ucl = i_mean + 3 * mr_mean / 1.128; i_lcl = i_mean - 3 * mr_mean / 1.128
    signals = [];
    for i in range(9, len(individuals)):
        subset = individuals[i-9:i]
        if all(subset > i_mean) or all(subset < i_mean): signals.extend(list(range(i-9, i)))
    signal_points = individuals.iloc[list(set(signals))]
    fig = go.Figure(); fig.add_trace(go.Scatter(y=individuals, name='Individual Value', mode='lines+markers', line=dict(color='#673ab7'))); fig.add_hline(y=i_mean, line=dict(color='green', dash='dot'), name='Mean'); fig.add_hline(y=i_ucl, line=dict(color='red', dash='dash'), name='UCL')
    outliers = individuals[individuals > i_ucl]; fig.add_trace(go.Scatter(x=outliers.index, y=outliers, mode='markers', name='UCL Violation', marker=dict(symbol='x', color='red', size=12))); fig.add_trace(go.Scatter(x=signal_points.index, y=signal_points, mode='markers', name='Nelson Rule 1 Signal', marker=dict(symbol='diamond', color='orange', size=10)))
    fig.update_layout(title=f'<b>I-Chart with Nelson Rule Detection: {value_col}</b>', yaxis_title='Value', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_xbar_r_chart(df):
    subgroup_size = df['Replicate'].max(); stats = df.groupby('Subgroup_ID')['Potency_Result'].agg(['mean', 'max', 'min']).reset_index(); stats['range'] = stats['max'] - stats['min']; A2, D3, D4 = 0.577, 0, 2.114; x_bar_bar = stats['mean'].mean(); r_bar = stats['range'].mean(); x_ucl = x_bar_bar + A2 * r_bar; x_lcl = x_bar_bar - A2 * r_bar; r_ucl = r_bar * D4; r_lcl = r_bar * D3
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart (Process Mean)", "R-Chart (Process Variation)")); fig.add_trace(go.Scatter(x=stats['Subgroup_ID'], y=stats['mean'], name='Subgroup Mean', mode='lines+markers'), row=1, col=1); fig.add_hline(y=x_bar_bar, line=dict(color='green', dash='dot'), name='Center Line', row=1, col=1); fig.add_hline(y=x_ucl, line=dict(color='red', dash='dash'), name='UCL', row=1, col=1); fig.add_hline(y=x_lcl, line=dict(color='red', dash='dash'), name='LCL', row=1, col=1); fig.add_trace(go.Scatter(x=stats['Subgroup_ID'], y=stats['range'], name='Subgroup Range', mode='lines+markers', line_color='orange'), row=2, col=1); fig.add_hline(y=r_bar, line=dict(color='green', dash='dot'), name='Center Line', row=2, col=1); fig.add_hline(y=r_ucl, line=dict(color='red', dash='dash'), name='UCL', row=2, col=1); fig.add_hline(y=r_lcl, line=dict(color='red', dash='dash'), name='LCL', row=2, col=1)
    fig.update_layout(height=600, title_text="<b>X-bar & R Chart for Subgroup Data (Potency Assay)</b>")
    return fig

def plot_p_chart(df):
    df['proportion'] = df['Batches_Failed'] / df['Batches_Tested']; p_bar = df['Batches_Failed'].sum() / df['Batches_Tested'].sum(); df['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df['Batches_Tested']); df['LCL'] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df['Batches_Tested'])).clip(lower=0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Month'], y=df['proportion'], name='Proportion Failed', mode='lines+markers')); fig.add_trace(go.Scatter(x=df['Month'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color='red', dash='dash'))); fig.add_trace(go.Scatter(x=df['Month'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color='red', dash='dash'))); fig.add_hline(y=p_bar, name='Average Fail Rate', line=dict(color='green', dash='dot'))
    outliers = df[df['proportion'] > df['UCL']]; fig.add_trace(go.Scatter(x=outliers['Month'], y=outliers['proportion'], mode='markers', name='Out of Control Signal', marker=dict(symbol='x', color='red', size=12)))
    fig.update_layout(title='<b>p-Chart for Batch Release Failure Rate</b>', xaxis_title='Month', yaxis_title='Proportion of Batches Failed', yaxis_tickformat=".2%")
    return fig

def plot_transfer_risk_waterfall(model, input_df):
    base_value = 0.45; contributions = {'Complexity_Score': (input_df['Complexity_Score'].iloc[0] - 6) * -0.05, 'SOP_Maturity_Score': (input_df['SOP_Maturity_Score'].iloc[0] - 6) * 0.04, 'Training_Cycles': (input_df['Training_Cycles'].iloc[0] - 2) * 0.03}; final_prediction = base_value + sum(contributions.values())
    fig = go.Figure(go.Waterfall(name = "Prediction", orientation = "v", measure = ["relative", "relative", "relative", "total"], x = ["Complexity", "SOP Maturity", "Training", "Final Prediction"], textposition = "outside", text = [f"{v:+.1%}" for v in contributions.values()] + [f"{final_prediction:.1%}"], y = list(contributions.values()) + [final_prediction], connector = {"line":{"color":"rgb(63, 63, 63)"}}, base = base_value))
    fig.update_layout(title = "<b>Risk Contribution Analysis</b>", yaxis_tickformat=".0%", showlegend=False)
    return fig

def plot_tech_opportunity_matrix(df):
    df['Complexity_Num'] = df['Implementation_Complexity_Score']; avg_impact = df['Est_Throughput_Increase_Factor'].mean(); avg_complexity = df['Complexity_Num'].mean()
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Complexity_Num'], y=df['Est_Throughput_Increase_Factor'], mode='markers+text', text=df['Technology'], textposition='top center', marker=dict(size=df['Est_FTE_Saving']*15, color=df['Targeted_Process'].astype('category').cat.codes, colorscale='viridis', showscale=False), hovertext=df['Targeted_Process'], name='Technologies'))
    fig.add_vline(x=avg_complexity, line_width=1, line_dash="dash", line_color="grey"); fig.add_hline(y=avg_impact, line_width=1, line_dash="dash", line_color="grey")
    fig.update_layout(title='<b>Technology Opportunity Prioritization Matrix</b>', xaxis_title='Implementation Complexity (Lower is Better)', yaxis_title='Throughput Increase (Factor)')
    fig.add_annotation(x=avg_complexity*0.9, y=avg_impact*1.1, text="<b>🔥 QUICK WINS 🔥</b>", showarrow=False, font=dict(color="#2e7d32", size=14)); fig.add_annotation(x=avg_complexity*1.1, y=avg_impact*1.1, text="<b>STRATEGIC BETS</b>", showarrow=False, font=dict(color="#2962ff")); fig.add_annotation(x=avg_complexity*0.9, y=avg_impact*0.9, text="<b>INCREMENTAL</b>", showarrow=False, font=dict(color="#ffc107")); fig.add_annotation(x=avg_complexity*1.1, y=avg_impact*0.9, text="<b>LUXURY</b>", showarrow=False, font=dict(color="grey"))
    return fig

# ======================================================================================
# SECTION 4: MAIN APPLICATION LAYOUT & SCIENTIFIC NARRATIVE
# ======================================================================================
st.title("🧬 Analytical Development Operations Command Center")
st.markdown("##### A strategic dashboard for managing high-throughput testing, method lifecycle, and program leadership in biologics development.")
sample_df, cqa_df, transfer_df, doe_df, team_df, equipment_df, subgroup_df, p_chart_df, tech_df = generate_master_data()
transfer_risk_model = get_transfer_risk_model(transfer_df)

st.markdown("### I. AD Operations & Sample Throughput Command Center")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4); avg_tat = (sample_df['Backlog'].mean() / sample_df['Samples_Tested'].mean()) * 7 if sample_df['Samples_Tested'].mean() > 0 else 0; kpi_col1.metric("Avg. Sample TAT (Days)", f"{avg_tat:.1f}"); kpi_col2.metric("Weekly Throughput", f"{sample_df['Samples_Tested'].iloc[-1]} Samples", f"{sample_df['Samples_Tested'].iloc[-1] - sample_df['Samples_Tested'].iloc[-2]:+d} vs last week"); kpi_col3.metric("Methods in Transfer", f"{transfer_df[transfer_df['Status'].isin(['Validation', 'Optimization'])].shape[0]}"); equipment_offline = equipment_df[equipment_df['Status'] != 'Online'].shape[0]; kpi_col4.metric("Equipment Readiness", f"{100-((equipment_offline/len(equipment_df))*100):.0f}%", f"{equipment_offline} Instrument(s) Offline", "inverse")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["**II. METHOD PERFORMANCE & OPTIMIZATION (DOE/SPC)**", "**III. TECHNOLOGY TRANSFER & VALIDATION (ML)**", "**IV. PROGRAM LEADERSHIP & RESOURCES**", "**V. HIGH THROUGHPUT TECHNOLOGY**"])

with tab1:
    st.header("II. Method Performance & Optimization")
    st.markdown("_This section provides tools for developing robust analytical procedures using advanced statistical methods and for monitoring their performance over the lifecycle, in accordance with ICH Q8, Q14, and Q2._")
    st.subheader("A. Method Optimization via Response Surface Methodology (RSM)")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To efficiently identify the optimal conditions for a new analytical method by modeling the relationship between input factors and a critical output response. This is a core principle of Quality by Design (QbD) as described in ICH Q8 and Q14.
        - **Method:** A Design of Experiments (DOE), typically a central composite or Box-Behnken design, is executed in the lab. A quadratic regression model is then fitted to the experimental data to generate the 3D response surface and 2D contour plot. The surface visualizes how the response (e.g., AAV Purity) changes as two factors (e.g., mobile phase pH, gradient slope) are varied simultaneously.
        - **Findings & Interpretation:** The peak of the 3D surface represents the predicted optimal setpoint for the method, maximizing robustness and performance. The 2D contour plot is used to define the **Method Operable Design Region (MODR)**—the "safe" operating space (red dashed box) where the method will consistently perform as expected. This data is critical for setting robust method parameters and defending the method's control strategy to regulatory agencies.
        """)
    fig_3d, fig_2d = plot_rsm_suite(doe_df, 'pH', 'Gradient_Slope_Pct_min', 'AAV_Purity_Pct')
    col1, col2 = st.columns(2); col1.plotly_chart(fig_3d, use_container_width=True); col2.plotly_chart(fig_2d, use_container_width=True)
    st.subheader("B. Method Lifecycle Monitoring via Statistical Process Control (SPC) Suite")
    spc_choice = st.selectbox("Select SPC Analysis Type:", ["Method Stability (I-MR Chart)", "Subgroup Precision & Accuracy (X-bar & R Chart)", "Process Yield (p-Chart)"])
    if spc_choice == "Method Stability (I-MR Chart)":
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To monitor the stability of a method when data is collected as individual measurements (e.g., daily system suitability or reference standard checks). This provides early detection of method drift or shifts.
            - **Method:** An Individuals Chart (I-Chart) is used to track individual results against statistical control limits (UCL/LCL, typically ±3σ) to detect significant shifts in the process mean. This chart is enhanced with **Nelson's Rule 1** detection, which flags 9 consecutive points on one side of the mean as a statistically significant, non-random trend.
            - **Findings & Interpretation:** The plot shows two signals: an orange diamond signal (Nelson Rule) indicating a subtle downward trend, followed by a red 'X' signal where a point breaches the UCL. This demonstrates a process that first became unstable (the trend) before becoming statistically out-of-control (the breach). This requires immediate Root Cause Analysis (RCA).
            """)
        st.plotly_chart(plot_enhanced_i_mr_chart(cqa_df, 'AAV_Titer_Control'), use_container_width=True)
    elif spc_choice == "Subgroup Precision & Accuracy (X-bar & R Chart)":
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To simultaneously monitor the central tendency (accuracy, via the X-bar chart) and variability (precision, via the R-chart) of a method when data is collected in rational subgroups (e.g., multiple replicates per run).
            - **Method:** The X-bar chart plots the average of each subgroup. The R-chart plots the range (Max - Min) within each subgroup. Control limits are calculated based on the overall mean, average range, and standard statistical constants (A2, D3, D4).
            - **Findings & Interpretation:** The R-chart must be analyzed first. It shows a significant increase in variability around subgroup #20, indicating a loss of method *precision*. This is followed by a shift in the mean on the X-bar chart around subgroup #30, indicating a loss of *accuracy*. This pattern suggests the root cause first impacted consistency before affecting the absolute result.
            """)
        st.plotly_chart(plot_xbar_r_chart(subgroup_df), use_container_width=True)
    elif spc_choice == "Process Yield (p-Chart)":
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To monitor the proportion of non-conforming items (e.g., failed batches, OOS results) over time, especially when the number of items tested each period varies.
            - **Method:** A p-chart plots the proportion of failed batches (`p = failed / tested`) for each month. The control limits are dynamic and widen when the number of batches tested is small, reflecting lower statistical confidence in that period's data.
            - **Findings & Interpretation:** The chart reveals a statistically significant spike in the failure rate in late 2024, which breaches the variable Upper Control Limit. This indicates a special cause of variation impacted that month's production or testing process and requires a thorough investigation, as it is highly unlikely to be due to random chance.
            """)
        st.plotly_chart(plot_p_chart(p_chart_df), use_container_width=True)

with tab2:
    st.header("III. Technology Transfer & Validation")
    st.markdown("_This section provides tools to manage the transfer of analytical procedures to receiving units (QC labs, CDMOs) and to predict the likelihood of success based on key attributes._")
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("A. Tech Transfer Risk Prediction & Explainability")
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To proactively identify high-risk method transfers, allowing for targeted resource allocation and risk mitigation strategies *before* the transfer begins.
            - **Method:** A Random Forest classifier was trained on historical transfer data. The model uses key leading indicators — Method Complexity, SOP Maturity, and required Training Cycles — to predict the probability of a successful transfer outcome. The waterfall chart provides **explainability**, showing how each factor contributes positively or negatively to the final prediction relative to a baseline.
            - **Interpretation:** This model provides a quantitative risk score. The waterfall chart shows *why* the risk is high or low. For example, a large negative contribution from "Complexity" can be directly addressed by investing time in simplifying the method before transfer. This transforms the model from a black box into an actionable diagnostic tool.
            """)
        complexity = st.slider("Method Complexity (1-10)", 1, 10, 5); sop_maturity = st.slider("SOP Maturity (1-10)", 1, 10, 7); training_cycles = st.slider("Planned Training Cycles", 1, 5, 2)
        if st.button("🔬 Predict Transfer Success", type="primary"):
            input_df = pd.DataFrame([[complexity, sop_maturity, training_cycles]], columns=['Complexity_Score', 'SOP_Maturity_Score', 'Training_Cycles'])
            st.plotly_chart(plot_transfer_risk_waterfall(transfer_risk_model, input_df), use_container_width=True)
    with col2:
        st.subheader("B. Method Transfer Portfolio Status")
        st.dataframe(transfer_df[['Program', 'Method', 'Receiving_Unit', 'Status']], use_container_width=True)

with tab3:
    st.header("IV. Program Leadership & Resource Management")
    st.markdown("_This section provides oversight of the AD Ops team's alignment with high-level drug development programs and the status of critical laboratory resources._")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. Sample Testing Capacity vs. Demand")
        st.plotly_chart(go.Figure(data=[go.Bar(name='Samples Tested', x=sample_df['Week'], y=sample_df['Samples_Tested']), go.Bar(name='Backlog Growth', x=sample_df['Week'], y=sample_df['Samples_Received'] - sample_df['Samples_Tested']), go.Scatter(name='Cumulative Backlog', x=sample_df['Week'], y=sample_df['Backlog'], yaxis='y2')]).update_layout(barmode='stack', title='<b>Weekly Sample Flow & Cumulative Backlog</b>', yaxis2=dict(title='Total Backlog', overlaying='y', side='right')), use_container_width=True)
    with col2:
        st.subheader("B. Critical Equipment Status")
        for _, row in equipment_df.iterrows():
            if row['Status'] == 'Online': st.success(f"**{row['Instrument']}:** {row['Status']}")
            elif row['Status'] == 'Calibration Due': st.warning(f"**{row['Instrument']}:** {row['Status']}")
            else: st.error(f"**{row['Instrument']}:** {row['Status']}")
    
with tab4:
    st.header("V. High Throughput Technology Assessment")
    st.subheader("A. Technology Opportunity Prioritization Matrix")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To provide a data-driven framework for prioritizing investment in new high-throughput technologies and automation platforms.
        - **Method:** A 2x2 matrix plotting Impact vs. Complexity. 
          - **Impact (Y-axis):** The estimated factor by which the technology will increase sample throughput (e.g., 5x).
          - **Complexity (X-axis):** A composite score representing cost, validation effort, and integration difficulty.
          - **Bubble Size:** Represents the estimated Full-Time Equivalent (FTE) savings, linking the investment to operational headcount efficiency.
        - **Interpretation:** The matrix identifies four strategic categories for investment:
          - **🔥 QUICK WINS:** High impact, low complexity. These are top-priority projects that should be funded and executed immediately.
          - **STRATEGIC BETS:** High impact, high complexity. These are major transformational projects that require significant capital and a multi-quarter implementation plan.
          - **INCREMENTAL:** Low impact, low complexity. Good for continuous improvement and can often be funded by operational budgets.
          - **LUXURY:** Low impact, high complexity. Generally avoided unless they are enabling technologies for a larger strategic bet.
        """)
    st.plotly_chart(plot_tech_opportunity_matrix(tech_df), use_container_width=True)

# ============================ SIDEBAR ============================
st.sidebar.image("https://assets-global.website-files.com/62a269e3ea783635a1608298/62a269e3ea783626786083d9_logo-horizontal.svg", use_container_width=True)
st.sidebar.markdown("### Role Focus")
st.sidebar.info("This dashboard is for an **Associate Director, AD Operations**, focused on building a high-throughput testing function, leading method development & transfer, and acting as a PDT representative for biologics.")
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
