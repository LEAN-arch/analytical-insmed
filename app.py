# ======================================================================================
# ANALYTICAL DEVELOPMENT OPERATIONS COMMAND CENTER
# v10.5 - Final, Fully Debugged & Robust Version
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import shap

# ======================================================================================
# SECTION 1: APP CONFIGURATION & AESTHETIC CONSTANTS
# ======================================================================================
st.set_page_config(
    page_title="AD Ops Command Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AESTHETIC & THEME CONSTANTS ---
PRIMARY_COLOR = '#673ab7'
SUCCESS_GREEN = '#4CAF50'
WARNING_AMBER = '#FFC107'
ERROR_RED = '#D32F2F'
NEUTRAL_GREY = '#B0BEC5'
DARK_GREY = '#455A64'
BACKGROUND_GREY = '#F0F2F6'
LIGHT_BLUE = '#E3F2FD'

# --- Custom CSS for professional look and feel ---
st.markdown(f"""
<style>
    .main .block-container {{ padding: 1rem 3rem 3rem; }}
    .stMetric {{ border-left: 5px solid {PRIMARY_COLOR}; }}
    h1, h2, h3 {{ color: {DARK_GREY}; }}
</style>
""", unsafe_allow_html=True)

# ======================================================================================
# SECTION 2: UTILITY & HELPER FUNCTIONS
# ======================================================================================
def render_manager_briefing(title: str, content: str, reg_refs: str, business_impact: str, quality_pillar: str, risk_mitigation: str) -> None:
    with st.container(border=True):
        st.subheader(f"🧬 {title}", divider='violet')
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}", icon="🎯")
        st.warning(f"**Key Guidelines & Regulations:** {reg_refs}", icon="📜")
        st.success(f"**Quality Culture Pillar:** {quality_pillar}", icon="🌟")
        st.error(f"**Strategic Risk Mitigation:** {risk_mitigation}", icon="🛡️")

def render_full_chart_briefing(context: str, significance: str, regulatory: str) -> None:
    briefing_card = f"""
    <div style="background-color: {LIGHT_BLUE}; border: 1px solid {PRIMARY_COLOR}; border-radius: 5px; padding: 15px; margin-bottom: 20px; color: {DARK_GREY};">
        <p style="margin-bottom: 10px;"><strong style="color: {PRIMARY_COLOR};">Context:</strong> {context}</p>
        <p style="margin-bottom: 10px;"><strong style="color: {DARK_GREY};">Significance:</strong> {significance}</p>
        <p style="margin-bottom: 0;"><strong style="color: {SUCCESS_GREEN};">Regulatory Alignment:</strong> {regulatory}</p>
    </div>
    """
    st.markdown(briefing_card, unsafe_allow_html=True)

def st_shap(plot, height=None):
    """Helper function to render SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# ======================================================================================
# SECTION 3: DATA GENERATION
# ======================================================================================
@st.cache_data(ttl=3600)
def generate_master_data():
    np.random.seed(42)
    # --- Base App Data ---
    budget_df = pd.DataFrame({'Category': ['OpEx (Consumables)', 'OpEx (Team)', 'CapEx (New Tech)', 'Contractors (CDMO)'], 'Budgeted': [800, 1200, 500, 300], 'Actual': [750, 1180, 550, 250]})
    team_df = pd.DataFrame({'Scientist': ['J. Doe (Lead)', 'S. Smith', 'M. Lee', 'K. Chen'], 'Role': ['Assoc. Director', 'Sr. Scientist', 'Scientist II', 'Scientist I'], 'Expertise': ['HPLC/CE', 'ddPCR/qPCR', 'Cell-Based Assays', 'ELISA'], 'Development_Goal': ['Mentor team on QbD', 'Lead AAV-201 analytics', 'Cross-train on ddPCR', 'Gain validation experience']})
    
    # --- Data for STATISTICAL TOOLKIT ---
    lj_data = np.random.normal(100.0, 2.0, 30); lj_data[20] = 106.5; lj_data[25:27] = [104.5, 104.8]; lj_df = pd.DataFrame({'Value': lj_data, 'Analyst': np.random.choice(['Smith', 'Lee'], 30)})
    ewma_data = np.random.normal(0.5, 0.05, 40); ewma_data[20:] += 0.06; ewma_df = pd.DataFrame({'Impurity': ewma_data, 'Batch': [f"AAV101-B{100+i}" for i in range(40)]})
    cusum_data = np.random.normal(10.0, 0.05, 50); cusum_data[25:] -= 0.04; cusum_df = pd.DataFrame({'Fill_Volume': cusum_data, 'Nozzle': np.random.choice([1,2,3,4], 50)})
    zone_data = np.random.normal(20, 0.5, 25); zone_data[15:] -= 0.4; zone_df = pd.DataFrame({'Seal_Strength': zone_data, 'Operator': np.random.choice(['Op-A', 'Op-B'], 25)})
    imr_data = np.random.normal(99.5, 0.1, 100); imr_data[80:] -= 0.25; imr_df = pd.DataFrame({'Purity': imr_data, 'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))})
    cpk_data = np.random.normal(50.5, 0.25, 150); cpk_df = pd.DataFrame({'Titer': cpk_data})
    
    # --- Hotelling T² data generation with fix ---
    mean_vec = [95, 1.0e13]
    std_devs = [1.5, 0.1e13]
    correlation = 0.7
    cov_mat_list = [[std_devs[0]**2, correlation * std_devs[0] * std_devs[1]], [correlation * std_devs[0] * std_devs[1], std_devs[1]**2]]
    cov_mat = (np.array(cov_mat_list) + np.array(cov_mat_list).T) / 2
    t2_data_in = np.random.multivariate_normal(mean_vec, cov_mat, 30)
    t2_outlier = [97.5, 0.9e13]; t2_data = np.vstack([t2_data_in[:24], t2_outlier, t2_data_in[24:]]); t2_df = pd.DataFrame(t2_data, columns=['Purity_Pct', 'Titer_vg_mL'])
    
    p_data = {'Month': pd.to_datetime(pd.date_range(start='2023-01-01', periods=12, freq='ME')), 'SSTs_Run': np.random.randint(40, 60, 12)}; p_df = pd.DataFrame(p_data); p_df['SSTs_Failed'] = np.random.binomial(n=p_df['SSTs_Run'], p=0.05); p_df.loc[9, 'SSTs_Failed'] = 8
    
    np_df = pd.DataFrame({'Week': range(1, 21), 'Batches_Sampled': 50, 'Defective_Vials': np.random.binomial(n=50, p=0.04, size=20)}); np_df.loc[12, 'Defective_Vials'] = 7
    c_df = pd.DataFrame({'Week': range(1, 21), 'Contaminants_per_Plate': np.random.poisson(lam=3, size=20)}); c_df.loc[15, 'Contaminants_per_Plate'] = 9
    u_data = {'Batch': range(1, 16), 'Vials_Inspected': np.random.randint(50, 150, 15)}; u_df = pd.DataFrame(u_data); u_df['Particulate_Defects'] = np.random.poisson(lam=u_df['Vials_Inspected'] * 0.02); u_df.loc[10, 'Particulate_Defects'] = 8
    
    # --- Data for LIFECYCLE HUB ---
    stability_df = pd.DataFrame({'Time_months': [0, 3, 6, 9, 12, 18, 24], 'Potency_pct': [101.2, 100.8, 99.5, 98.9, 98.1, 97.0, 95.8]})
    tost_df = pd.DataFrame({'HPLC': np.random.normal(98.5, 0.5, 30), 'UPLC': np.random.normal(98.7, 0.4, 30)})
    screening_df = pd.DataFrame({'Factor': ['Temp', 'pH', 'Flow_Rate', 'Gradient', 'Column_Lot', 'Analyst'], 'Effect_Size': [0.2, 1.8, 0.1, 1.5, 0.3, 0.2]})
    doe_df = pd.DataFrame(np.random.uniform(-1, 1, (15, 2)), columns=['pH', 'Gradient_Slope']); doe_df['Peak_Resolution'] = 2.5 - 0.5*doe_df['pH']**2 - 0.8*doe_df['Gradient_Slope']**2 + 0.3*doe_df['pH']*doe_df['Gradient_Slope'] + np.random.normal(0, 0.1, 15)
    
    # --- Data for PREDICTIVE HUB ---
    oos_df = pd.DataFrame({'Instrument': np.random.choice(['HPLC-01', 'HPLC-02', 'CE-01'], 100), 'Analyst': np.random.choice(['Smith', 'Lee', 'Chen'], 100), 'Molecule_Type': np.random.choice(['mAb', 'AAV'], 100), 'Root_Cause': np.random.choice(['Sample_Prep_Error', 'Instrument_Malfunction', 'Column_Issue'], 100, p=[0.5, 0.3, 0.2])})
    
    # Use `.clip(min=...)` for NumPy arrays.
    backlog_vals = 10 + np.arange(104)*0.5 + np.random.normal(0, 5, 104) + np.sin(np.arange(104)/8)*5
    backlog_df = pd.DataFrame({'Week': pd.date_range('2022-01-01', periods=104, freq='W'), 'Backlog': backlog_vals.clip(min=0)})
    
    maintenance_df = pd.DataFrame({'Run_Hours': np.random.randint(50, 1000, 100), 'Pressure_Spikes': np.random.randint(0, 20, 100), 'Column_Age_Days': np.random.randint(10, 300, 100)}); maintenance_df['Needs_Maint'] = (maintenance_df['Run_Hours'] > 600) | (maintenance_df['Pressure_Spikes'] > 15) | (maintenance_df['Column_Age_Days'] > 250)

    # --- Data for QBD & QUALITY SYSTEMS HUB ---
    sankey_df = pd.DataFrame({'Source': ['Column Lot', 'Mobile Phase Purity', 'Gradient Slope', 'Flow Rate', 'Column Temp', 'Peak Resolution', 'Assay Accuracy'], 'Target': ['Peak Resolution', 'Peak Resolution', 'Peak Resolution', 'Assay Precision', 'Assay Accuracy', 'Final Purity Result', 'Final Purity Result'], 'Value': [8, 5, 10, 7, 6, 12, 10]})

    return budget_df, team_df, lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df, stability_df, tost_df, screening_df, doe_df, oos_df, backlog_df, maintenance_df, sankey_df

# ======================================================================================
# SECTION 4: PLOTTING & ANALYSIS FUNCTIONS
# ======================================================================================
## --- STATISTICAL TOOLKIT FUNCTIONS (ENRICHED) ---
def plot_levey_jennings(df):
    render_full_chart_briefing(context="Daily QC analysis of a certified reference material on an HPLC system to ensure system performance.", significance="Detects shifts or increased variability in an analytical instrument's performance, ensuring the validity of daily sample results. It distinguishes between random error (a single outlier) and systematic error (a developing bias).", regulatory="Directly supports **21 CFR 211.160** (General requirements for laboratory controls) and **ISO 17025** by providing documented evidence of the ongoing validity of test methods. Westgard rules are an industry best practice for clinical and QC labs.")
    mean, sd = 100.0, 2.0
    fig = go.Figure()
    fig.add_hrect(y0=mean - 3*sd, y1=mean + 3*sd, line_width=0, fillcolor='rgba(255, 193, 7, 0.1)', layer="below", name='±3s Zone')
    fig.add_hrect(y0=mean - 2*sd, y1=mean + 2*sd, line_width=0, fillcolor='rgba(76, 175, 80, 0.2)', layer="below", name='±2s Zone')
    fig.add_hrect(y0=mean - sd, y1=mean + sd, line_width=0, fillcolor='rgba(76, 175, 80, 0.3)', layer="below", name='±1s Zone')
    df['Deviation_SD'] = (df['Value'] - mean) / sd
    fig.add_trace(go.Scatter(y=df['Value'], mode='lines+markers', line=dict(color=PRIMARY_COLOR), customdata=df[['Analyst', 'Deviation_SD']], hovertemplate='<b>Value: %{y:.2f}</b><br>Analyst: %{customdata[0]}<br>Deviation: %{customdata[1]:.2f} SD<extra></extra>'))
    fig.add_hline(y=mean, line_dash='solid', line_color=SUCCESS_GREEN, annotation_text="Mean")
    fig.add_annotation(x=20, y=106.5, text="<b>1-3s Violation</b><br>Random Error", showarrow=True, arrowhead=2, ax=0, ay=-50, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.add_annotation(x=26, y=104.8, text="<b>2-2s Violation</b><br>Systematic Bias", showarrow=True, arrowhead=2, ax=0, ay=50, bgcolor=WARNING_AMBER)
    fig.update_layout(title="<b>Levey-Jennings Chart with Westgard Rule Violations</b>", yaxis_title="Reference Material Recovery (%)", xaxis_title="Run Number")
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The 1-3s violation indicates a random error, requiring a re-run. The subsequent 2-2s violation indicates a systematic bias. **Decision:** Halt testing on the instrument and issue a work order to investigate the mobile phase preparation process and recalibrate the detector.")

def plot_ewma_chart(df):
    render_full_chart_briefing(context="Monitoring a critical quality attribute (CQA), like the concentration of a key impurity, over dozens of purification runs.", significance="Provides a highly sensitive early warning for small but persistent process drifts that could lead to out-of-specification (OOS) results if left unchecked, enabling proactive rather than reactive maintenance.", regulatory="Supports the principles of Continued Process Verification (CPV) outlined in the **FDA's Process Validation Guidance** and **ICH Q8**. It is a tool to ensure the process remains in a state of control throughout its lifecycle.")
    lam = 0.2; mean = df['Impurity'].iloc[:20].mean(); sd = df['Impurity'].iloc[:20].std(); df['EWMA'] = df['Impurity'].ewm(span=(2/lam)-1, adjust=False).mean(); n = df.index + 1; ucl = mean + 3 * sd * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * n))); lcl = mean - 3 * sd * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * n)))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['Impurity'], mode='markers', name='Individual Batch', marker_color=NEUTRAL_GREY, customdata=df['Batch'], hovertemplate='Batch: %{customdata}<br>Impurity: %{y:.3f}%<extra></extra>')); fig.add_trace(go.Scatter(x=df.index, y=df['EWMA'], mode='lines', name='EWMA', line=dict(color=PRIMARY_COLOR, width=3))); fig.add_trace(go.Scatter(x=df.index, y=ucl, mode='lines', name='UCL', line=dict(color=ERROR_RED, dash='dash'))); fig.add_trace(go.Scatter(x=df.index, y=lcl, mode='lines', name='LCL', line=dict(color=ERROR_RED, dash='dash')))
    violation_idx = df[df['EWMA'] > ucl].first_valid_index(); fig.add_annotation(x=violation_idx, y=df['EWMA'][violation_idx], text="<b>EWMA Signal: Process Drift Detected</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title="<b>EWMA Chart: Early Detection of Impurity Drift</b>", yaxis_title="Impurity Level (%)", xaxis_title="Batch Number")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The EWMA chart signaled an out-of-control condition at Batch #{violation_idx}, much earlier than a standard chart would have. This early warning confirms a slow degradation of the purification column. **Decision:** Schedule the column for repacking during the next planned maintenance window, preventing future OOS results.")

def plot_cusum_chart(df):
    render_full_chart_briefing(context="Monitoring a critical process parameter like fill volume on a high-speed aseptic filling line.", significance="Allows for the fastest possible detection of the *onset* of a small, sustained process shift. This is critical for minimizing the amount of non-conforming product generated in high-volume, high-cost manufacturing processes.", regulatory="This is an advanced SPC tool that demonstrates a mature quality system focused on rapid response, aligning with the risk-based principles of **ICH Q9**. It provides a higher level of process control than basic Shewhart charts.")
    target = 10.0; sd = 0.05; k = 0.5 * sd; h = 5 * sd; df['SH-'] = 0.0
    for i in range(1, len(df)): df.loc[i, 'SH-'] = max(0, df.loc[i-1, 'SH-'] + target - df.loc[i, 'Fill_Volume'] - k)
    fig = go.Figure(); fig.add_trace(go.Scatter(y=df['SH-'], name='CUSUM Low (SH-)', mode='lines+markers', line=dict(color=PRIMARY_COLOR, width=3), customdata=df, hovertemplate='<b>Sample %{x}</b><br>Nozzle: %{customdata[1]}<br>CUSUM Value: %{y:.3f}<extra></extra>')); fig.add_hline(y=h, line_dash='dash', line_color=ERROR_RED, annotation_text="Decision Limit (H)")
    violation_idx = df[df['SH-'] > h].first_valid_index(); fig.add_annotation(x=violation_idx, y=df['SH-'][violation_idx], text="<b>CUSUM Signal!</b>", showarrow=True, arrowhead=2, ax=20, ay=-60, bgcolor=PRIMARY_COLOR, font=dict(color='white'))
    fig.update_layout(title="<b>CUSUM Chart: Rapid Detection of Fill Volume Shift</b>", yaxis_title="Cumulative Sum from Target", xaxis_title="Sample Number")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The CUSUM chart rapidly detected a persistent downward shift, signaling an alarm at Sample #{violation_idx}. This confirmed an underfill condition had begun. **Decision:** The filling line was immediately halted. Investigation traced the issue to a partial nozzle clog, saving the majority of the batch from being non-conforming.")

def plot_zone_chart(df):
    render_full_chart_briefing(context="Monitoring the stability of a mature, well-understood, and highly capable analytical method, like a validated potency assay.", significance="Detects unnatural, non-random patterns *within* the control limits. It provides a much earlier warning of potential process drift than a standard chart that only alarms on UCL/LCL breaches, allowing for proactive investigation.", regulatory="Demonstrates a sophisticated level of process understanding and monitoring, aligning with the principles of **ICH Q14** (Analytical Procedure Development) and Continued Process Verification. The use of sensitizing rules (e.g., Nelson, Westgard) is a hallmark of a mature quality system.")
    mean, sd = 20, 0.5; fig = go.Figure()
    zones = {'Zone A (Upper)': [mean + 2*sd, mean + 3*sd], 'Zone B (Upper)': [mean + 1*sd, mean + 2*sd], 'Zone C (Upper)': [mean, mean + 1*sd], 'Zone C (Lower)': [mean - 1*sd, mean], 'Zone B (Lower)': [mean - 2*sd, mean - 1*sd], 'Zone A (Lower)': [mean - 3*sd, mean - 2*sd]}
    colors = {'Zone A (Upper)': 'rgba(255, 193, 7, 0.2)', 'Zone B (Upper)': 'rgba(76, 175, 80, 0.2)', 'Zone C (Upper)': 'rgba(76, 175, 80, 0.2)', 'Zone C (Lower)': 'rgba(76, 175, 80, 0.2)', 'Zone B (Lower)': 'rgba(255, 193, 7, 0.2)', 'Zone A (Lower)': 'rgba(255, 193, 7, 0.2)'}
    for name, y_range in zones.items(): fig.add_hrect(y0=y_range[0], y1=y_range[1], line_width=0, fillcolor=colors[name], annotation_text=name.split(' ')[1], annotation_position="top left", layer="below")
    fig.add_trace(go.Scatter(y=df['Seal_Strength'], mode='lines+markers', name='Strength', line=dict(color=PRIMARY_COLOR), customdata=df['Operator'], hovertemplate='Strength: %{y:.2f} N<br>Operator: %{customdata}<extra></extra>')); fig.add_hline(y=mean, line_color='black')
    fig.add_annotation(x=18, y=mean - 0.7, text="<b>Rule Violation!</b><br>8 consecutive points<br>on one side of mean.", showarrow=False, bgcolor=WARNING_AMBER, borderpad=4)
    fig.update_layout(title="<b>Zone Chart for Potency Assay Control with Sensitizing Rules</b>", yaxis_title="Relative Potency (%)", xaxis_title="Assay Run")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Actionable Insight:** Although no single point is out of control, the Zone Chart detected a run of 8 consecutive points below the center line. This non-random pattern indicates a systematic process shift. **Decision:** This early warning triggers an investigation into the stability of the cell bank or critical reagents used in the assay during the next planned maintenance cycle.")

def plot_i_mr_chart(df):
    render_full_chart_briefing(context="Monitoring individual measurements of a critical process parameter over time, such as the purity of a reference standard checked daily.", significance="Provides a fundamental assessment of process stability by monitoring both the process mean (I-chart) and its short-term variability (MR-chart). A spike in variability is often the first sign of a problem.", regulatory="This is a foundational SPC tool. Its use is fundamental to demonstrating a state of statistical control as required by the **FDA's Process Validation Guidance** and for setting meaningful alert limits in a laboratory environment (**21 CFR 211.160, 211.165**).")
    i = df['Purity']; i_mean = i.mean(); mr = abs(i.diff()); mr_mean = mr.mean(); i_ucl = i_mean + 2.66 * mr_mean; i_lcl = i_mean - 2.66 * mr_mean; mr_ucl = 3.267 * mr_mean
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart:</b> Process Mean", "<b>Moving Range (MR) Chart:</b> Process Variability"))
    fig.add_trace(go.Scatter(x=df['Date'], y=i, name='Purity', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1); fig.add_hline(y=i_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1); fig.add_hline(y=i_ucl, line_dash="dot", line_color=ERROR_RED, row=1, col=1); fig.add_hline(y=i_lcl, line_dash="dot", line_color=ERROR_RED, row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=mr, name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1); fig.add_hline(y=mr_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1); fig.add_hline(y=mr_ucl, line_dash="dot", line_color=ERROR_RED, row=2, col=1)
    fig.update_layout(height=500, showlegend=False, title_text="<b>I-MR Chart for Reference Standard Purity</b>")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Actionable Insight:** The I-chart shows a clear downward trend and multiple points breaching the lower control limit, indicating the reference standard is no longer stable and is degrading. **Decision:** Quarantine the current standard vial and qualify a new one to ensure continued data accuracy for all ongoing tests.")

def plot_cpk_analysis(df):
    render_full_chart_briefing(context="Assessing a final, validated manufacturing process to ensure it can reliably produce a drug substance that meets its critical quality attribute specifications.", significance="Quantifies the *capability* of a process, answering 'Is our process good enough?'. It measures how well the process is centered within its specification limits and how much variability it has relative to those limits. A Cpk ≥ 1.33 is a common industry standard for a capable process.", regulatory="Cpk is a key metric in Process Validation (Stage 2) and Continued Process Verification (Stage 3) per the **FDA's Process Validation Guidance**. It provides objective evidence that a process is capable of consistently producing quality product, a core tenet of **ICH Q8 (QbD)**.")
    data = df['Titer']; LSL, USL, target = 48.0, 52.0, 50.0; mu, std = data.mean(), data.std(ddof=1)
    cpk, cp, pp, ppk = 0,0,0,0
    if std > 0: cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std)); cp = (USL - LSL) / (6*std); pp = (USL-LSL)/(6*np.sqrt(np.mean((data-mu)**2))); ppk = min((USL-mu)/(3*np.sqrt(np.mean((data-mu)**2))), (mu-LSL)/(3*np.sqrt(np.mean((data-mu)**2))))
    
    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure(go.Histogram(x=data, name='Observed Data', histnorm='probability density', marker_color=PRIMARY_COLOR, opacity=0.7)); x_fit = np.linspace(data.min(), data.max(), 200); y_fit = stats.norm.pdf(x_fit, mu, std); fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Normal', line=dict(color=SUCCESS_GREEN, width=2)))
        fig.add_vline(x=LSL, line_dash="dash", line_color=ERROR_RED, annotation_text="LSL=48"); fig.add_vline(x=USL, line_dash="dash", line_color=ERROR_RED, annotation_text="USL=52"); fig.add_vline(x=target, line_dash="dot", line_color=DARK_GREY, annotation_text="Target=50")
        fig.update_layout(title_text=f'<b>Process Capability Analysis - Cpk = {cpk:.2f} (Capable)</b>', xaxis_title="Titer Result (e12 vg/mL)", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Capability Indices")
        st.metric("Cpk (Process Capability)", f"{cpk:.2f}", "Target: >= 1.33")
        st.metric("Ppk (Process Performance)", f"{ppk:.2f}", "Target: >= 1.33")
        st.metric("Cp (Process Potential)", f"{cp:.2f}")
        st.metric("Pp (Performance Potential)", f"{pp:.2f}")
        st.info("Cpk measures short-term (within-subgroup) variation, while Ppk measures long-term, overall variation.")
    if cpk >= 1.33: st.success(f"**Actionable Insight:** All capability indices, especially the critical Cpk of {cpk:.2f}, exceed the target of 1.33. This statistically proves the process is robust, centered, and capable of consistently meeting its specification. **Decision:** The process is validated and approved for commercial production.")
    else: st.error(f"**Actionable Insight:** The Cpk of {cpk:.2f} is below the required 1.33. The process is not capable. **Decision:** Process requires re-development to reduce variability or re-center the mean before re-validation.")

def plot_hotelling_t2_chart(df):
    render_full_chart_briefing(context="Simultaneously monitoring two critical, correlated drug substance attributes, like Purity and Titer, from a bioreactor run.", significance="Prevents 'hiding in the noise'. A small, simultaneous shift in two correlated variables might not trigger an alarm on either individual chart, but the T² chart detects these joint-shifts, preventing the release of a non-compliant batch.", regulatory="This demonstrates a mature understanding of process control, aligning with **ICH Q8**'s emphasis on multivariate interactions. It is a key tool for implementing a Process Analytical Technology (PAT) or Multivariate Statistical Process Control (MSPC) program.")
    t_squared_values = np.random.chisquare(2, size=len(df)) * 0.8; ucl = 9.21; t_squared_values[24] = 15.5
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_t2 = go.Figure(go.Scatter(y=t_squared_values, mode='lines+markers', name='T² Value', line=dict(color=PRIMARY_COLOR))); fig_t2.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text="UCL"); fig_t2.add_annotation(x=24, y=15.5, text="<b>Multivariate Anomaly!</b>", showarrow=True, bgcolor=ERROR_RED, font=dict(color='white'))
        fig_t2.update_layout(title="<b>Hotelling's T² Chart</b>", yaxis_title="T² Statistic", xaxis_title="Batch Number")
        st.plotly_chart(fig_t2, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(df, x='Purity_Pct', y='Titer_vg_mL', title="<b>Process Variable Correlation</b>"); fig_scatter.add_trace(go.Scatter(x=[df['Purity_Pct'].iloc[24]], y=[df['Titer_vg_mL'].iloc[24]], mode='markers', marker=dict(color=ERROR_RED, size=12, symbol='x'), name='Anomaly'))
        fig_scatter.update_shapes(type='circle', xref='x', yref='y', x0=df['Purity_Pct'].mean()-2*df['Purity_Pct'].std(), y0=df['Titer_vg_mL'].mean()-2*df['Titer_vg_mL'].std(), x1=df['Purity_Pct'].mean()+2*df['Purity_Pct'].std(), y1=df['Titer_vg_mL'].mean()+2*df['Titer_vg_mL'].std(), line_color=PRIMARY_COLOR, opacity=0.3)
        st.plotly_chart(fig_scatter, use_container_width=True)
    st.error("**Actionable Insight:** The T² chart identified a multivariate anomaly at Batch #25. The scatter plot with the 95% confidence ellipse shows why: while Purity was only slightly high and Titer only slightly low, their combination was far outside the normal process correlation, a failure mode two separate charts would have missed. **Decision:** Quarantine Batch #25 and launch an investigation into the raw materials used.")

def plot_p_chart(df):
    render_full_chart_briefing(context="Monitoring the proportion of monthly HPLC System Suitability Tests (SSTs) that fail.", significance="Tracks the failure rate of a key quality system when the number of tests performed each month varies. It provides a high-level view of the health and reliability of an entire analytical system.", regulatory="Directly supports quality system monitoring as required by **21 CFR 211** and **EudraLex Vol. 4**. Tracking SST failures is a critical component of laboratory control and data integrity (**ALCOA+**).")
    df['proportion'] = df['SSTs_Failed'] / df['SSTs_Run']; p_bar = df['SSTs_Failed'].sum() / df['SSTs_Run'].sum(); df['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run']); 
    df['LCL'] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run'])).clip(lower=0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Month'], y=df['proportion'], name='Proportion Failed', mode='lines+markers')); fig.add_trace(go.Scatter(x=df['Month'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'))); fig.add_hline(y=p_bar, name='Average Fail Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    fig.update_layout(title='<b>p-Chart for System Suitability Test (SST) Failure Rate</b>', yaxis_title='Proportion of SSTs Failed', yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The p-chart reveals a statistically significant spike in the SST failure rate in October, breaching the UCL. This indicates a special cause of variation. **Decision:** Launch an investigation focused on events in October. Review all column changes, mobile phase preparations, and instrument maintenance records from that period to find the root cause.")

def plot_np_chart(df):
    render_full_chart_briefing(context="Tracking the number of rejected vials with cosmetic defects per batch, where each batch inspection always consists of exactly 50 vials.", significance="Provides a simple, direct way to monitor the number of defective items when the sample size is constant. It is often easier for operators to understand and react to than a proportion-based chart.", regulatory="A fundamental tool for lot acceptance and release testing programs (**21 CFR 211.165**). It provides an ongoing record of quality for in-process controls or final product inspection.")
    # FIX: Split assignment to resolve UnboundLocalError. 'n' must be defined before it is used.
    n = df['Batches_Sampled'].iloc[0]
    p_bar = df['Defective_Vials'].sum() / (len(df) * n)
    ucl = n * p_bar + 3 * np.sqrt(n * p_bar * (1-p_bar))
    lcl = max(0, n * p_bar - 3 * np.sqrt(n * p_bar * (1-p_bar)))
    fig = go.Figure(go.Scatter(x=df['Week'], y=df['Defective_Vials'], mode='lines+markers')); fig.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=lcl, line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=n*p_bar, line_dash='dot', line_color=SUCCESS_GREEN)
    fig.add_annotation(x=12, y=7, text="<b>Spike Detected</b>", bgcolor=ERROR_RED, font_color='white')
    fig.update_layout(title='<b>np-Chart for Number of Defective Vials per Batch</b>', yaxis_title='Count of Defective Vials (n=50)', xaxis_title='Week')
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** A statistically significant spike in defective vials was detected in Week 13. **Decision:** Place the associated lot on hold. Launch a root cause investigation focusing on the vial washing and siliconization process for that specific production run.")

def plot_c_chart(df):
    render_full_chart_briefing(context="Monitoring the number of bioburden colonies found per environmental monitoring (EM) plate placed in a critical Grade A area.", significance="Tracks the level of microbiological control in a cleanroom. An out-of-control signal is a direct indicator of a potential contamination event that could impact product sterility and patient safety.", regulatory="Directly supports compliance with **EudraLex Vol. 4, Annex 1** (Manufacture of Sterile Medicinal Products) and **ISO 14644**. It provides objective evidence that the classified environment is being maintained in a state of control as required by **21 CFR 211.113**.")
    c_bar = df['Contaminants_per_Plate'].mean(); ucl = c_bar + 3 * np.sqrt(c_bar); lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
    fig = go.Figure(go.Scatter(x=df['Week'], y=df['Contaminants_per_Plate'], mode='lines+markers')); fig.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=lcl, line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=c_bar, line_dash='dot', line_color=SUCCESS_GREEN)
    fig.add_annotation(x=15, y=9, text="<b>Spike Detected</b>", bgcolor=ERROR_RED, font_color='white')
    fig.update_layout(title='<b>c-Chart for Environmental Monitoring (Bioburden)</b>', yaxis_title='Colony Count per Plate', xaxis_title='Week')
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The c-chart shows a statistically significant spike in the colony count during Week 16. This indicates a special cause event. **Decision:** Place the room on hold and launch an immediate investigation, reviewing cleaning logs, personnel access, and HVAC performance data for that week.")

def plot_u_chart(df):
    render_full_chart_briefing(context="Monitoring the rate of particulate defects found during the visual inspection of finished drug product vials, where the number of vials inspected from each batch varies.", significance="Provides a normalized measure of quality (defects per unit) that is comparable across batches of different sizes. This is crucial for accurately assessing process stability when production volumes fluctuate.", regulatory="A more sophisticated tool for lot release and stability testing (**21 CFR 211.165, 211.166**). Using a u-chart over a simpler c-chart demonstrates a higher level of statistical understanding when dealing with variable sample sizes.")
    df['defects_per_unit'] = df['Particulate_Defects'] / df['Vials_Inspected']; u_bar = df['Particulate_Defects'].sum() / df['Vials_Inspected'].sum(); df['UCL'] = u_bar + 3 * np.sqrt(u_bar / df['Vials_Inspected']);
    df['LCL'] = (u_bar - 3 * np.sqrt(u_bar / df['Vials_Inspected'])).clip(lower=0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Batch'], y=df['defects_per_unit'], name='Defect Rate', mode='lines+markers')); fig.add_trace(go.Scatter(x=df['Batch'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'))); fig.add_hline(y=u_bar, name='Average Defect Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    fig.add_annotation(x=11, y=df['defects_per_unit'].iloc[10], text="<b>Spike Detected</b>", bgcolor=ERROR_RED, font_color='white')
    fig.update_layout(title='<b>u-Chart for Particulate Defect Rate per Vial</b>', yaxis_title='Defects per Vial', xaxis_title='Batch Number')
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The u-chart detected a statistically significant increase in the defect rate for Batch 11, even though the absolute number of defects might not have seemed alarming on its own. **Decision:** Quarantine Batch 11. The investigation should focus on the glass vial supplier and the washing process specific to that production run.")

## --- LIFECYCLE HUB FUNCTIONS ---
def plot_stability_analysis(df):
    render_full_chart_briefing(context="Analyzing long-term stability data for an AAV drug product to determine its shelf-life.", significance="Provides the statistical justification for a product's expiration date, a critical component of any regulatory submission and a key patient safety consideration.", regulatory="Directly follows the methodology outlined in **ICH Q1E: Evaluation of Stability Data**. It requires statistical analysis to propose a retest period or shelf life.")
    spec_limit = 90.0; time_points = df['Time_months']; potency = df['Potency_pct']; slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, potency)
    df['Fit'] = slope * df['Time_months'] + intercept; df['Residual'] = df['Potency_pct'] - df['Fit']; t_inv = stats.t.ppf(0.975, len(df)-2); s_err = np.sqrt(np.sum(df['Residual']**2) / (len(df)-2))
    df['CI_Upper'] = df['Fit'] + t_inv * s_err * np.sqrt(1/len(df) + (df['Time_months'] - df['Time_months'].mean())**2 / np.sum((df['Time_months'] - df['Time_months'].mean())**2))
    df['CI_Lower'] = df['Fit'] - t_inv * s_err * np.sqrt(1/len(df) + (df['Time_months'] - df['Time_months'].mean())**2 / np.sum((df['Time_months'] - df['Time_months'].mean())**2))
    fig = px.scatter(df, x='Time_months', y='Potency_pct', title="<b>ICH Q1E Stability Analysis & Shelf-Life Estimation</b>"); fig.add_trace(go.Scatter(x=df['Time_months'], y=df['Fit'], name='Regression', line=dict(color='red'))); fig.add_trace(go.Scatter(x=df['Time_months'], y=df['CI_Upper'], name='95% CI', line=dict(color='red', dash='dash'), showlegend=False)); fig.add_trace(go.Scatter(x=df['Time_months'], y=df['CI_Lower'], name='95% CI', fill='tonexty', line=dict(color='red', dash='dash'), fillcolor='rgba(255,0,0,0.1)'))
    fig.add_hline(y=spec_limit, line_dash='dash', line_color=ERROR_RED, annotation_text="Specification Limit (90%)")
    shelf_life = (spec_limit - intercept) / slope if slope < 0 else "N/A"
    if isinstance(shelf_life, float): fig.add_vline(x=shelf_life, line_dash='dot', line_color=SUCCESS_GREEN, annotation_text=f"Est. Shelf Life: {shelf_life:.1f} mo")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The linear regression analysis of the stability data, including the 95% confidence interval for the degradation rate, predicts a shelf-life of **{shelf_life:.1f} months**. **Decision:** Propose a 24-month shelf life for the AAV-101 drug product in the upcoming BLA submission, providing this data as statistical justification.")

def render_doe_suite(screening_df, doe_df):
    render_full_chart_briefing(context="Developing a new HPLC purity method from first principles, following Quality by Design (QbD).", significance="A strategic, two-phase approach to method development. The Screening phase efficiently identifies the few critical parameters ('vital few') from the many potential ones ('trivial many'). The Optimization phase then precisely models the behavior of those critical parameters to define a robust operating space.", regulatory="This structured approach is the core principle of **ICH Q8(R2)** and **ICH Q14**. It moves beyond trial-and-error, creating a deep process understanding that is highly valued by regulatory agencies.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phase 1: Screening Design")
        fig_pareto = px.bar(screening_df.sort_values('Effect_Size', ascending=False), x='Factor', y='Effect_Size', title="<b>Screening: Identifying Critical Parameters</b>", labels={'Effect_Size': 'Standardized Effect on Resolution'})
        fig_pareto.add_hline(y=1.0, line_dash='dash', line_color=ERROR_RED, annotation_text="Significance Threshold")
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.info("The Pareto plot shows that **pH** and **Gradient Slope** have statistically significant effects on peak resolution. The other factors can be fixed at standard levels.")
    with col2:
        st.subheader("Phase 2: Optimization (RSM)")
        feature_names = ['pH', 'Gradient_Slope']
        poly = PolynomialFeatures(degree=2); X_poly = poly.fit_transform(doe_df[feature_names]); model = LinearRegression().fit(X_poly, doe_df['Peak_Resolution'])
        x = np.linspace(-1, 1, 30); y = np.linspace(-1, 1, 30); x_grid, y_grid = np.meshgrid(x, y)
        
        grid_df = pd.DataFrame(np.c_[x_grid.ravel(), y_grid.ravel()], columns=feature_names)
        X_pred_poly = poly.transform(grid_df)
        
        z_grid = model.predict(X_pred_poly).reshape(x_grid.shape)
        fig_rsm = go.Figure(data=[go.Contour(z=z_grid, x=x, y=y, colorscale='viridis', contours=dict(coloring='heatmap', showlabels=True))]); fig_rsm.add_shape(type="rect", x0=-0.5, y0=-0.7, x1=0.5, y1=0.7, line=dict(color="white", width=3, dash="dot"), fillcolor="rgba(255,255,255,0.3)"); fig_rsm.add_annotation(x=0, y=0, text="<b>MODR</b>", font=dict(color='white', size=16), showarrow=False); fig_rsm.update_layout(title="<b>Optimization: Defining the MODR</b>", xaxis_title="pH (Normalized)", yaxis_title="Gradient Slope (Normalized)")
        st.plotly_chart(fig_rsm, use_container_width=True)
        st.info("The contour plot defines the Method Operable Design Region (MODR). The method setpoint will be chosen at the center of this space for maximum robustness.")
    st.success("**Actionable Insight:** The two-phase DOE approach provides a highly efficient and scientifically sound path to a robust method. **Decision:** Finalize the method SOP with the setpoints derived from the RSM optimization and proceed to formal validation.")

def plot_method_equivalency_tost(df):
    render_full_chart_briefing(context="Comparing a newly developed, faster UPLC method against the legacy validated HPLC method to ensure results are comparable before replacing the old method.", significance="Proves that two methods are statistically equivalent and can be used interchangeably. This is critical for post-approval changes, method updates, or transfers between sites without needing to re-establish all specifications.", regulatory="The TOST (Two One-Sided T-Tests) approach is the gold standard for demonstrating equivalency and is preferred by the **FDA** over simple t-tests. It directly supports lifecycle management as described in **ICH Q12**.")
    diff = df['UPLC'] - df['HPLC']; n = len(diff); mean_diff = diff.mean(); std_diff = diff.std(ddof=1); se_diff = std_diff / np.sqrt(n); t_crit = stats.t.ppf(0.95, df=n-1); ci_lower = mean_diff - t_crit * se_diff; ci_upper = mean_diff + t_crit * se_diff; equiv_limit = 0.5
    fig = go.Figure(); fig.add_trace(go.Scatter(x=diff, y=np.zeros_like(diff), mode='markers', marker=dict(color=NEUTRAL_GREY, opacity=0.5), name='Paired Differences')); fig.add_trace(go.Scatter(x=[mean_diff], y=[0], mode='markers', marker=dict(color=PRIMARY_COLOR, size=15, symbol='diamond'), error_x=dict(type='data', array=[ci_upper - mean_diff], arrayminus=[mean_diff - ci_lower], thickness=4), name='90% CI of Mean Difference'))
    fig.add_vline(x=-equiv_limit, line_dash='dash', line_color=ERROR_RED, annotation_text="Lower Equiv. Limit"); fig.add_vline(x=equiv_limit, line_dash='dash', line_color=ERROR_RED, annotation_text="Upper Equiv. Limit"); fig.update_yaxes(visible=False)
    fig.update_layout(title="<b>Method Equivalency via TOST</b>", xaxis_title="Difference in Purity (%) [UPLC - HPLC]")
    st.plotly_chart(fig, use_container_width=True)
    if ci_lower > -equiv_limit and ci_upper < equiv_limit: st.success("**Actionable Insight:** The 90% confidence interval for the mean difference is completely contained within the pre-defined equivalence limits of ±0.5%. **Decision:** The new UPLC method is statistically equivalent to the legacy HPLC method. A protocol can be drafted to replace the old method in the QC lab.")
    else: st.error("**Actionable Insight:** The confidence interval crosses one of the equivalence limits, failing to prove equivalence. **Decision:** The methods cannot be used interchangeably. An investigation into the source of the bias between the two methods is required.")

## --- PREDICTIVE HUB FUNCTIONS ---
@st.cache_resource
def get_oos_rca_model(_df):
    df_encoded = pd.get_dummies(_df, columns=['Instrument', 'Analyst', 'Molecule_Type'])
    features = [col for col in df_encoded.columns if col != 'Root_Cause']; target = 'Root_Cause'; X, y = df_encoded[features], df_encoded[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y); return model, X.columns

def run_oos_prediction_model(df):
    render_full_chart_briefing(context="An analyst reports an Out-of-Specification (OOS) result and a formal lab investigation is initiated.", significance="This predictive tool acts as a 'digital SME' to guide the investigation. By analyzing the parameters of the failed run, it suggests the most probable root cause, focusing the initial troubleshooting efforts and saving valuable time.", regulatory="Demonstrates a proactive, data-driven approach to laboratory investigations as required by **21 CFR 211.192**. It ensures investigations are thorough, timely, and based on historical data patterns, not just guesswork.")
    model, feature_cols = get_oos_rca_model(df)
    col1, col2, col3 = st.columns(3); instrument = col1.selectbox("Instrument Used", df['Instrument'].unique()); analyst = col2.selectbox("Analyst", df['Analyst'].unique()); molecule = col3.selectbox("Molecule Type", df['Molecule_Type'].unique())
    if st.button("🔬 Predict Probable Root Cause", type="primary"):
        input_data = pd.DataFrame(0, columns=feature_cols, index=[0])
        input_data[f'Instrument_{instrument}'] = 1
        input_data[f'Analyst_{analyst}'] = 1
        input_data[f'Molecule_Type_{molecule}'] = 1
        pred_proba = model.predict_proba(input_data); proba_df = pd.DataFrame(pred_proba.T, index=model.classes_, columns=['Probability']).sort_values('Probability', ascending=False)
        fig = px.bar(proba_df, x='Probability', y=proba_df.index, orientation='h', title="<b>Predicted Root Cause Probability</b>", text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**Actionable Insight:** The model predicts the highest probability root cause is **'{proba_df.index[0]}'.** **Decision:** The lab investigation should begin by immediately focusing on this area. For example, if 'Sample_Prep_Error' is indicated, the first step is to interview the analyst and review their specific sample preparation records.")

def plot_backlog_forecast(df):
    render_full_chart_briefing(context="The AD Ops leader needs to plan resource allocation (headcount, instrument time) for the upcoming quarters.", significance="Moves planning from reactive to proactive. By forecasting the future sample backlog, a leader can provide data-driven justification for hiring new staff or purchasing new equipment *before* the lab becomes a bottleneck to the entire R&D organization.", regulatory="While not a direct compliance requirement, this demonstrates strong resource and capacity planning, a key competency for laboratory management under quality systems like **ISO 17025** and general GxP.")
    
    model = SimpleExpSmoothing(df['Backlog'], initialization_method="estimated").fit()
    forecast = model.forecast(26) # Forecast 6 months (26 weeks)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Week'], y=df['Backlog'], name='Historical Backlog', line=dict(color=PRIMARY_COLOR)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecasted Backlog', line=dict(color=ERROR_RED, dash='dash')))
    
    st.plotly_chart(fig, use_container_width=True)
    st.error(f"**Actionable Insight:** The time series forecast predicts that the sample backlog will exceed **{forecast.iloc[-1]:.0f} samples** within the next 6 months. This trend is unsustainable with the current staffing level. **Decision:** Submit a formal headcount request for one additional Research Associate, using this forecast as the primary data-driven justification.")

@st.cache_resource
def get_maint_model(_df):
    features = ['Run_Hours', 'Pressure_Spikes', 'Column_Age_Days']; target = 'Needs_Maint'; X, y = _df[features], _df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y); return model

def run_hplc_maintenance_model(df):
    render_full_chart_briefing(context="Managing a fleet of heavily used HPLC instruments and deciding which ones to prioritize for preventative maintenance.", significance="Shifts maintenance from a fixed, time-based schedule to a proactive, condition-based model. It reduces unnecessary maintenance on healthy instruments and prevents unexpected failures on high-risk instruments, maximizing uptime.", regulatory="Ensures instruments remain in a qualified state as required by **21 CFR 211.160(b)**. A predictive model provides a sophisticated, risk-based approach (**ICH Q9**) to maintaining the validated state of equipment.")
    model = get_maint_model(df)
    col1, col2, col3 = st.columns(3); hours = col1.slider("Total Run Hours", 50, 1000, 750); spikes = col2.slider("Pressure Spikes >100psi", 0, 20, 18); age = col3.slider("Column Age (Days)", 10, 300, 280)
    input_df = pd.DataFrame([[hours, spikes, age]], columns=['Run_Hours', 'Pressure_Spikes', 'Column_Age_Days']); pred_prob = model.predict_proba(input_df)[0][1]
    
    colA, colB = st.columns([1,2])
    with colA:
        fig = go.Figure(go.Indicator(mode = "gauge+number", value = pred_prob * 100, title = {'text': "Maintenance Risk Score"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': ERROR_RED if pred_prob > 0.7 else WARNING_AMBER if pred_prob > 0.4 else SUCCESS_GREEN}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.subheader("Explainable AI (XAI): Why this score?")
        st.info("This SHAP plot shows which factors are pushing the risk score higher (red) or lower (blue). The size of the bar indicates the magnitude of the factor's impact.")
        explainer = shap.TreeExplainer(model)
        
        # FIX: Use the more robust SHAP object-oriented pattern to avoid TypeErrors.
        # This creates an Explanation object that bundles all necessary components.
        shap_explanation = explainer(input_df)
        
        # Pass the first (and only) explanation object from the batch to the force_plot.
        st_shap(shap.force_plot(shap_explanation[0]), height=150)

    st.warning("**Actionable Insight:** The model predicts a very high probability that HPLC-01 requires preventative maintenance. The SHAP analysis reveals that the high number of **Run Hours** and **Pressure Spikes** are the primary drivers of this risk score. **Decision:** Schedule HPLC-01 for maintenance this week, prioritizing it over other instruments with lower risk scores to prevent an unexpected failure during a critical run.")

## --- QBD & QUALITY SYSTEMS HUB FUNCTIONS ---
def render_qbd_sankey_chart(df):
    render_full_chart_briefing(context="Defining the relationships between material attributes, process parameters, and quality attributes for an HPLC purity method.", significance="This visualizes the core of QbD: understanding and controlling the linkages between what goes into a process (CMAs), what the process does (CPPs), and the quality of the output (CQAs). It forms the basis of a robust control strategy.", regulatory="This is a direct visual representation of the principles outlined in **ICH Q8 (Pharmaceutical Development)**. It provides clear justification for the parameters chosen for validation and routine monitoring.")
    all_nodes = pd.unique(df[['Source', 'Target']].values.ravel('K')); node_map = {node: i for i, node in enumerate(all_nodes)}
    fig = go.Figure(data=[go.Sankey(node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=PRIMARY_COLOR), link = dict(source = df['Source'].map(node_map), target = df['Target'].map(node_map), value = df['Value']))])
    fig.update_layout(title_text="<b>QbD Control Strategy: Linking CMAs/CPPs to CQAs</b>", font_size=12)
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The diagram clearly shows that 'Gradient Slope' is a Critical Process Parameter as it strongly influences the 'Peak Resolution' CQA. Therefore, this parameter must have a tight control range in the final method and SOP. 'Flow Rate', which only impacts precision, can have a wider acceptable range.")

def render_method_v_model():
    render_full_chart_briefing(context="Establishing a formal, traceable development plan for a new analytical method before any lab work begins.", significance="The V-Model ensures that development is systematic and that every requirement is ultimately verified. It prevents scope creep and ensures the final, validated method is guaranteed to be fit for its intended purpose.", regulatory="This visualizes the Design Control process, a fundamental concept in **21 CFR 820.30** (for medical devices, but a best practice for pharma) and **ISO 13485**. It ensures a direct link between user needs (ATP) and validated performance.")
    fig = go.Figure(); fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["<b>Analytical Target Profile (ATP)</b>", "<b>Method Requirements</b>", "<b>Method Design (e.g., Column Choice)</b>", "<b>Method Procedure (SOP Draft)</b>"], textposition="bottom center", line=dict(color=PRIMARY_COLOR, width=3), marker=dict(size=15))); fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["<b>Procedure Verification</b>", "<b>Instrument Qualification (IQ/OQ)</b>", "<b>Method Validation (ICH Q2)</b>", "<b>ATP Confirmation (Fitness for Use)</b>"], textposition="top center", line=dict(color=SUCCESS_GREEN, width=3), marker=dict(size=15)))
    links = [("ATP ↔ Fitness for Use", 4), ("Requirements ↔ Validation", 3), ("Design ↔ Qualification", 2), ("Procedure ↔ Verification", 1)]
    for i, (text, y_pos) in enumerate(links): fig.add_shape(type="line", x0=4-i, y0=y_pos, x1=5+i, y1=y_pos, line=dict(color=NEUTRAL_GREY, width=1, dash="dot")); fig.add_annotation(x=4.5, y=y_pos + 0.1, text=text, showarrow=False)
    fig.update_layout(title_text="<b>Analytical Method Design Control (V-Model)</b>", title_x=0.5, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The V-model provides a clear roadmap for the entire team. By defining the validation criteria upfront based on the ATP, we de-risk the project and ensure that our final validation exercise will definitively prove the method is fit-for-purpose.")

def run_interactive_rca_fishbone():
    render_full_chart_briefing(context="An Out-of-Specification (OOS) result for purity has been confirmed, and a formal laboratory investigation has been launched.", significance="A structured RCA tool like a Fishbone diagram prevents 'tunnel vision' during an investigation. It forces the team to consider a wide range of potential causes across different categories, leading to a more thorough and accurate root cause determination.", regulatory="Using a formal RCA tool is a best practice for OOS investigations, as required by **21 CFR 211.192**. It provides documented, objective evidence that the investigation was systematic and not just a cursory check.")
    
    st.subheader("Fishbone Diagram for OOS Investigation")
    st.info("This tool structures the brainstorming process for a root cause analysis investigation.")
    
    causes = {
        'Man (Analyst)': ['Inadequate training', 'Improper sample preparation', 'Calculation error'],
        'Machine (Instrument)': ['Leaky pump seal', 'Detector lamp aging', 'Incorrect injection volume'],
        'Material (Reagents/Sample)': ['Reference standard degraded', 'Contaminated mobile phase', 'Sample degradation'],
        'Method': ['SOP is unclear or incorrect', 'Method not robust to lab conditions', 'Incorrect column equilibration time'],
        'Measurement': ['Integration parameters incorrect', 'Calibration curve expired', 'Wrong standard concentration used'],
        'Environment': ['Lab temperature out of range', 'Vibration near balance', 'Power fluctuation']
    }
    
    cols = st.columns(3)
    categories = list(causes.keys())
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{cat}**")
                for cause in causes[cat]:
                    st.markdown(f"- {cause}")

    st.success("**Actionable Insight:** The investigation team uses this structured tool to brainstorm. After testing several hypotheses, the team confirmed through re-analysis with a freshly prepared standard that the **'Reference standard degraded'** (under 'Material') was the true root cause. **Decision:** A CAPA will be initiated to revise the reference standard management SOP to include more frequent stability checks.")

def render_troubleshooting_flowchart():
    st.info("This flowchart provides a systematic, compliant path for investigating an Out-of-Specification (OOS) result.")
    graph_definition = """
    digraph OOS_Flowchart {
        rankdir=TB;
        node [shape=box, style="rounded,filled", fillcolor="#E3F2FD", color="#673ab7", fontname="sans-serif"];
        edge [color="#455A64", fontname="sans-serif"];

        subgraph cluster_phase1 {
            label = "Phase 1: Initial Investigation";
            style="rounded,filled";
            color="#F0F2F6";
            
            oos [label="OOS Result Confirmed"];
            check [label="Lab Investigation: Obvious Error Check\\n(e.g., calculation, dilution)"];
            error_found [shape=diamond, label="Obvious Error Found?"];
            invalidate [label="Invalidate Result (with justification)\\nRe-test per SOP"];
            no_error [label="No Obvious Error Found"];

            oos -> check -> error_found;
            error_found -> invalidate [label="Yes"];
            error_found -> no_error [label="No"];
        }
        
        subgraph cluster_phase2 {
            label = "Phase 2: Full-Scale Investigation";
            style="rounded,filled";
            color="#F0F2F6";

            full_rca [label="Conduct Full RCA\\n(Fishbone, 5 Whys)"];
            mfg [label="Review Manufacturing & Process Data"];
            retest [label="Hypothesis-Driven Retesting\\n(e.g., new column, fresh reagents)"];
            root_cause [shape=diamond, label="Root Cause Identified?"];
            
            no_error -> full_rca;
            full_rca -> mfg;
            full_rca -> retest;
        }
        
        subgraph cluster_phase3 {
            label = "Phase 3: Conclusion & CAPA";
            style="rounded,filled";
            color="#F0F2F6";
            
            capa [label="Implement CAPA\\n(Corrective & Preventive Action)"];
            batch_decision [label="Make Batch Disposition Decision\\n(Release, Reject, Rework)"];
            conclude [label="Close Investigation"];
            
            root_cause -> capa [label="Yes"];
            root_cause -> batch_decision [label="No (Inconclusive)"];
            capa -> batch_decision;
            batch_decision -> conclude;
        }

        retest -> root_cause;
        mfg -> root_cause;
    }
    """
    st.graphviz_chart(graph_definition)
    st.success("**Actionable Insight:** An OOS investigation must be a formal, documented process. This flowchart ensures all required steps are taken, from the initial check for simple errors to a full-scale RCA and CAPA implementation. **Decision:** All lab personnel will be retrained on this OOS procedure to ensure consistent and compliant execution.")

# ======================================================================================
# SECTION 5: PAGE RENDERING FUNCTIONS
# ======================================================================================
def render_strategic_hub_page(budget_df, team_df):
    st.title("Executive & Strategic Hub")
    render_manager_briefing(title="Leading Analytical Development as a High-Impact Business Unit", content="This hub demonstrates the strategic, business-oriented aspects of leading the AD Ops function. It covers financial management, goal setting (OKRs), and, most importantly, the development and mentorship of the scientific team.", reg_refs="Company HR Policies, Departmental Budget", business_impact="Ensures the department operates in a fiscally responsible manner, is aligned with corporate goals, and fosters a culture of growth and expertise.", quality_pillar="Leadership & People Management.", risk_mitigation="Prevents skill gaps on critical projects and justifies resource needs through clear, data-driven forecasting and performance tracking.")
    st.subheader("Departmental OKRs", divider='violet'); st.dataframe(pd.DataFrame({"Objective": ["Accelerate PD Support", "Enhance Method Robustness", "Foster Team Growth"], "Key Result": ["Reduce avg. sample TAT by 15%", "Implement QbD for 2 new methods", "Cross-train 2 scientists on ddPCR"], "Status": ["On Track", "Complete", "On Track"]}).style.map(lambda s: f"background-color: {SUCCESS_GREEN if s in ['On Track', 'Complete'] else WARNING_AMBER}; color: white;"), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1: st.subheader("Annual Budget Performance", divider='violet'); st.plotly_chart(px.bar(budget_df, x='Category', y=budget_df['Actual'] - budget_df['Budgeted']), use_container_width=True)
    with col2: st.subheader("Team Skill & Development Matrix", divider='violet'); st.dataframe(team_df, use_container_width=True, hide_index=True); st.success("**Actionable Insight:** The matrix identifies a potential bottleneck in cell-based assays. M. Lee will be assigned as the lead for the next potency assay development project, mentored by J. Doe, to support their growth.")

def render_statistical_toolkit_page(lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df):
    st.title("Advanced Statistical Toolkit")
    render_manager_briefing(title="Applying Statistical Rigor to Analytical Problems", content="This hub serves as a comprehensive toolkit, demonstrating deep, first-principles expertise in applying the correct statistical process control (SPC) and capability tools to different analytical challenges. Each chart is presented with a realistic, domain-specific context to show not just *how* to perform the analysis, but *why* and *when* to use it.", reg_refs="ICH Q9 (Quality Risk Management), 21 CFR 211.165(d) (Statistics)", business_impact="Ensures that decisions about process control, method performance, and product quality are based on objective statistical evidence, not intuition.", quality_pillar="Statistical Thinking & Data Literacy.", risk_mitigation="Detects process drifts, out-of-control states, and capability issues early, preventing large-scale failures, batch rejections, and invalid data.")
    tab1, tab2, tab3 = st.tabs(["**📊 Monitoring Process Stability & Drift**", "**📈 Monitoring Quality & Yield (Attribute Data)**", "**🔎 Advanced Process & Method Insights**"])
    with tab1: st.subheader("Tools for Monitoring Continuous Data", divider='violet'); plot_i_mr_chart(imr_df); plot_levey_jennings(lj_df); plot_ewma_chart(ewma_df); plot_cusum_chart(cusum_df); plot_zone_chart(zone_df)
    with tab2: st.subheader("Tools for Monitoring Attribute (Count/Fail) Data", divider='violet'); plot_p_chart(p_df); plot_np_chart(np_df); plot_c_chart(c_df); plot_u_chart(u_df)
    with tab3: st.subheader("Tools for Deeper Process Understanding", divider='violet'); plot_hotelling_t2_chart(t2_df); plot_cpk_analysis(cpk_df)

def render_lifecycle_hub_page(stability_df, tost_df, screening_df, doe_df):
    st.title("Method & Product Lifecycle Hub")
    render_manager_briefing(title="Guiding Methods from R&D to Commercial Launch", content="This hub demonstrates the strategic oversight of the entire analytical and product lifecycle. It showcases a deep understanding of early-stage method development (QbD), late-stage product stability, and the statistical tools needed to manage post-approval changes and method transfers.", reg_refs="ICH Q1E (Stability), ICH Q12 (Lifecycle Management), ICH Q8/Q14 (QbD)", business_impact="Creates robust, well-understood analytical methods that are less prone to failure, accelerates development timelines, and provides a solid data foundation for regulatory filings and product shelf-life justification.", quality_pillar="Lifecycle Management & Scientific Rigor.", risk_mitigation="Front-loads risk management into the development phase, preventing costly validation failures and ensuring methods are transferable and maintainable throughout the product's commercial life.")
    st.subheader("Early-Stage: Quality by Design (QbD) for Method Development", divider='violet'); render_doe_suite(screening_df, doe_df)
    st.subheader("Late-Stage: Commercial & Post-Approval Support", divider='violet'); plot_stability_analysis(stability_df); plot_method_equivalency_tost(tost_df)

def render_predictive_hub_page(oos_df, backlog_df, maintenance_df):
    st.title("Predictive Operations & Diagnostics")
    render_manager_briefing(title="Building a Proactive, Data-Driven Operations Function", content="This hub showcases a forward-looking leadership approach, using predictive analytics and machine learning to move the AD Ops function from a reactive service center to a proactive, strategic partner. These tools are used to forecast future challenges, diagnose problems faster, and optimize resource allocation.", reg_refs="ICH Q9 (Quality Risk Management), FDA's Computer Software Assurance (CSA) Guidance", business_impact="Maximizes instrument uptime, accelerates OOS investigations, and provides data-driven justification for resource planning, ultimately increasing the speed and efficiency of the entire R&D organization.", quality_pillar="Predictive Analytics & Continuous Improvement.", risk_mitigation="Anticipates future bottlenecks, equipment failures, and quality issues, allowing for mitigation *before* they occur and impact critical program timelines.")
    st.subheader("Predictive Diagnostics & Troubleshooting", divider='violet'); run_oos_prediction_model(oos_df)
    st.subheader("Proactive Resource & Maintenance Planning", divider='violet'); plot_backlog_forecast(backlog_df); run_hplc_maintenance_model(maintenance_df)

def render_qbd_quality_systems_hub_page(sankey_df):
    st.title("QbD & Quality Systems Hub")
    render_manager_briefing(title="Integrating Quality Systems into Analytical Development", content="This hub demonstrates a deep understanding of modern quality systems and philosophies. It showcases how to proactively build quality into methods from the start using Quality by Design (QbD) and Design Controls, and how to react to deviations with systematic, compliant problem-solving tools.", reg_refs="ICH Q8, Q9, Q10; 21 CFR 820.30 (Design Controls); 21 CFR 211.192 (Investigations)", business_impact="Creates fundamentally more robust and reliable methods, reduces validation failures, streamlines regulatory submissions, and ensures investigations are efficient and scientifically sound.", quality_pillar="Proactive Quality & Systematic Problem Solving.", risk_mitigation="Moves the function from a 'test-and-fix' mentality to a 'design-and-understand' paradigm, fundamentally de-risking the entire method lifecycle and ensuring compliance with formal investigation requirements.")
    st.subheader("Proactive Quality by Design (QbD) & Design Controls", divider='violet'); render_qbd_sankey_chart(sankey_df); render_method_v_model()
    st.subheader("Reactive Problem Solving & Root Cause Analysis (RCA)", divider='violet'); run_interactive_rca_fishbone(); render_troubleshooting_flowchart()

# ======================================================================================
# SECTION 6: MAIN APP ROUTER (SIDEBAR NAVIGATION)
# ======================================================================================
st.sidebar.title("AD Ops Navigation")
PAGES = {
    "Executive & Strategic Hub": render_strategic_hub_page,
    "QbD & Quality Systems Hub": render_qbd_quality_systems_hub_page,
    "Method & Product Lifecycle Hub": render_lifecycle_hub_page,
    "Predictive Operations & Diagnostics": render_predictive_hub_page,
    "Advanced Statistical Toolkit": render_statistical_toolkit_page,
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Retrieve Data ---
(
    budget_df, team_df, lj_df, ewma_df, cusum_df, zone_df, imr_df, 
    cpk_df, t2_df, p_df, np_df, c_df, u_df, stability_df, tost_df, 
    screening_df, doe_df, oos_df, backlog_df, maintenance_df, sankey_df
) = generate_master_data()

# --- Render the selected page ---
page_function = PAGES[selection]

if selection == "Executive & Strategic Hub":
    page_function(budget_df, team_df)
elif selection == "QbD & Quality Systems Hub":
    page_function(sankey_df)
elif selection == "Method & Product Lifecycle Hub":
    page_function(stability_df, tost_df, screening_df, doe_df)
elif selection == "Predictive Operations & Diagnostics":
    page_function(oos_df, backlog_df, maintenance_df)
elif selection == "Advanced Statistical Toolkit":
    page_function(lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df)

st.sidebar.markdown("---")
st.sidebar.markdown("### Role Focus")
st.sidebar.info("This portfolio is for an **Associate Director, Analytical Development Operations** role, demonstrating leadership in building high-throughput testing functions, managing the method lifecycle, and applying advanced statistical methods.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Key Regulatory & Quality Frameworks")
with st.sidebar.expander("View Applicable Guidelines", expanded=False):
    st.markdown("- **ICH Q1E, Q2, Q8, Q9, Q10, Q12, Q14**\n- **21 CFR Parts 11, 211, 820.30**\n- **EudraLex Vol. 4, Annex 1 & 15**\n- **ISO 17025, ISO 13485**")
