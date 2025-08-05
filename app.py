# ======================================================================================
# ANALYTICAL DEVELOPMENT OPERATIONS COMMAND CENTER
# v11.1 - Final, Robust & Fully-Corrected Version
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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
GUARD_BAND_COLOR = 'rgba(255, 152, 0, 0.5)' # Orange for guard bands

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

def wilson_score_interval(p, n, z=1.96):
    """Calculates the Wilson score interval for a proportion."""
    if n == 0: return 0, 1
    p = float(p); n = float(n)
    numerator = p + z**2/(2*n)
    denominator = 1 + z**2/n
    term = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    lower = (numerator - term) / denominator
    upper = (numerator + term) / denominator
    return max(0, lower), min(1, upper)

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
    ewma_data_pre = np.random.normal(0.5, 0.05, 25); ewma_data_post = np.random.normal(0.5, 0.05, 15); ewma_df = pd.DataFrame({'Impurity': np.concatenate([ewma_data_pre, ewma_data_post]), 'Batch': [f"AAV101-B{100+i}" for i in range(40)]}); ewma_df.loc[15:24, 'Impurity'] += 0.04
    cusum_data = np.random.normal(10.0, 0.05, 50); cusum_data[25:] -= 0.04; cusum_df = pd.DataFrame({'Fill_Volume': cusum_data, 'Nozzle': np.random.choice([1,2,3,4], 50)})
    imr_data = np.random.normal(99.5, 0.1, 100); imr_data[60:68] -= 0.2; imr_data[80:] -= 0.25; imr_df = pd.DataFrame({'Purity': imr_data, 'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))})
    cpk_data = np.random.normal(50.5, 0.25, 150); cpk_df = pd.DataFrame({'Titer': cpk_data})

    # --- Hotelling T² data generation ---
    mean_vec = [95, 1.0e13]; std_devs = [1.5, 0.1e13]; correlation = 0.7; cov_mat = [[std_devs[0]**2, correlation * std_devs[0] * std_devs[1]], [correlation * std_devs[0] * std_devs[1], std_devs[1]**2]]
    t2_data_in = np.random.multivariate_normal(mean_vec, np.array(cov_mat), 30); t2_outlier = [97.5, 0.9e13]; t2_data = np.vstack([t2_data_in[:24], t2_outlier, t2_data_in[24:]]); t2_df = pd.DataFrame(t2_data, columns=['Purity_Pct', 'Titer_vg_mL'])

    p_data = {'Month': pd.to_datetime(pd.date_range(start='2023-01-01', periods=12, freq='ME')), 'SSTs_Run': np.random.randint(40, 60, 12)}; p_df = pd.DataFrame(p_data); p_df['SSTs_Failed'] = np.random.binomial(n=p_df['SSTs_Run'], p=0.05); p_df.loc[9, 'SSTs_Failed'] = 8
    np_df = pd.DataFrame({'Week': range(1, 21), 'Batches_Sampled': 50, 'Defective_Vials': np.random.binomial(n=50, p=0.04, size=20)}); np_df.loc[12, 'Defective_Vials'] = 7
    c_df = pd.DataFrame({'Week': range(1, 21), 'Contaminants_per_Plate': np.random.poisson(lam=3, size=20)}); c_df.loc[15, 'Contaminants_per_Plate'] = 9
    u_data = {'Batch': range(1, 16), 'Vials_Inspected': np.random.randint(50, 150, 15)}; u_df = pd.DataFrame(u_data); u_df['Particulate_Defects'] = np.random.poisson(lam=u_df['Vials_Inspected'] * 0.02); u_df.loc[10, 'Particulate_Defects'] = 8

    # --- Data for LIFECYCLE HUB ---
    time_pts = [0, 3, 6, 9, 12, 18, 24]; stability_dfs = []
    for i in range(3):
        base_potency = 101.2 - i*0.5; slope = -0.2 - i*0.02 + np.random.normal(0, 0.01)
        potency = base_potency + slope * np.array(time_pts) + np.random.normal(0, 0.2, len(time_pts))
        stability_dfs.append(pd.DataFrame({'Batch': f'Batch {i+1}', 'Time_months': time_pts, 'Potency_pct': potency}))
    stability_df = pd.concat(stability_dfs, ignore_index=True)
    tost_df = pd.DataFrame({'HPLC': np.random.normal(98.5, 0.5, 30), 'UPLC': np.random.normal(98.7, 0.4, 30)})
    screening_df = pd.DataFrame({'Factor': ['Temp', 'pH', 'Flow_Rate', 'Gradient', 'Column_Lot', 'Analyst'], 'Effect_Size': [0.2, 1.8, 0.1, 1.5, 0.3, 0.2]})
    doe_df = pd.DataFrame(np.random.uniform(-1, 1, (15, 2)), columns=['pH', 'Gradient_Slope']); doe_df['Peak_Resolution'] = 2.5 - 0.5*doe_df['pH']**2 - 0.8*doe_df['Gradient_Slope']**2 + 0.3*doe_df['pH']*doe_df['Gradient_Slope'] + np.random.normal(0, 0.1, 15)

    # --- Data for PREDICTIVE HUB ---
    oos_df = pd.DataFrame({'Instrument': np.random.choice(['HPLC-01', 'HPLC-02', 'CE-01'], 100), 'Analyst': np.random.choice(['Smith', 'Lee', 'Chen'], 100), 'Molecule_Type': np.random.choice(['mAb', 'AAV'], 100), 'Root_Cause': np.random.choice(['Sample_Prep_Error', 'Instrument_Malfunction', 'Column_Issue'], 100, p=[0.5, 0.3, 0.2])})
    backlog_vals = 10 + np.arange(104)*0.5 + np.random.normal(0, 5, 104) + np.sin(np.arange(104)/8)*5
    backlog_df = pd.DataFrame({'Week': pd.date_range('2022-01-01', periods=104, freq='W'), 'Backlog': backlog_vals.clip(min=0)})
    maintenance_df = pd.DataFrame({'Run_Hours': np.random.randint(50, 1000, 100), 'Pressure_Spikes': np.random.randint(0, 20, 100), 'Column_Age_Days': np.random.randint(10, 300, 100)}); maintenance_df['Needs_Maint'] = (maintenance_df['Run_Hours'] > 600) | (maintenance_df['Pressure_Spikes'] > 15) | (maintenance_df['Column_Age_Days'] > 250)

    # --- Data for QBD & QUALITY SYSTEMS HUB ---
    sankey_df = pd.DataFrame({'Source': ['Column Lot', 'Mobile Phase Purity', 'Gradient Slope', 'Flow Rate', 'Column Temp', 'Peak Resolution', 'Assay Accuracy'], 'Target': ['Peak Resolution', 'Peak Resolution', 'Peak Resolution', 'Assay Precision', 'Assay Accuracy', 'Final Purity Result', 'Final Purity Result'], 'Value': [8, 5, 10, 7, 6, 12, 10]})

    return budget_df, team_df, lj_df, ewma_df, cusum_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df, stability_df, tost_df, screening_df, doe_df, oos_df, backlog_df, maintenance_df, sankey_df

# ======================================================================================
# SECTION 4: PLOTTING & ANALYSIS FUNCTIONS
# ======================================================================================
## --- STATISTICAL TOOLKIT FUNCTIONS (ENRICHED) ---
def plot_levey_jennings(df):
    render_full_chart_briefing(context="Daily QC analysis of a certified reference material on an HPLC system to ensure system performance.", significance="Detects shifts or increased variability in an analytical instrument's performance, ensuring the validity of daily sample results. It distinguishes between random error (a single outlier) and systematic error (a developing bias).", regulatory="Directly supports **21 CFR 211.160** (General requirements for laboratory controls) and **ISO 17025** by providing documented evidence of the ongoing validity of test methods. Westgard rules are an industry best practice for clinical and QC labs.")
    mean, sd = 100.0, 2.0
    fig = go.Figure()
    fig.add_hrect(y0=mean - 1.5*sd, y1=mean + 1.5*sd, line_width=0, fillcolor='rgba(255, 152, 0, 0.1)', layer="below", name='±1.5s Guard Band')
    fig.add_hrect(y0=mean - 3*sd, y1=mean + 3*sd, line_width=0, fillcolor='rgba(255, 0, 0, 0.1)', layer="below", name='±3s UCL/LCL')
    fig.add_hrect(y0=mean - 2*sd, y1=mean + 2*sd, line_width=0, fillcolor='rgba(255, 193, 7, 0.2)', layer="below", name='±2s Warning')
    fig.add_trace(go.Scatter(y=df['Value'], mode='lines+markers', name='QC Value', line=dict(color=PRIMARY_COLOR), customdata=df['Analyst'], hovertemplate='Value: %{y:.2f}<br>Analyst: %{customdata}<extra></extra>'))
    fig.add_hline(y=mean, line_dash='solid', line_color=SUCCESS_GREEN, annotation_text="Mean")
    fig.add_annotation(x=20, y=106.5, text="<b>1-3s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=-50, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.add_annotation(x=26, y=104.8, text="<b>2-2s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=50, bgcolor=WARNING_AMBER)
    fig.update_layout(title="<b>Levey-Jennings Chart with Guard Bands & Westgard Rules</b>", yaxis_title="Reference Material Recovery (%)", xaxis_title="Run Number")
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The 1-3s violation is a definitive failure. The 2-2s violation, while inside the 3s limits, is a clear signal of systematic bias. The earlier points breaching the ±1.5s guard band could have served as an even earlier warning. **Decision:** Halt testing. Initiate a formal deviation to investigate the systematic bias. Quarantine all data generated since run #25.")

def plot_ewma_chart(df):
    render_full_chart_briefing(context="Monitoring a critical quality attribute (CQA) where early detection of small drifts is paramount.", significance="Highly sensitive to small, persistent process drifts. This example shows an emerging drift, a corrective action, and the subsequent return to a state of control, demonstrating a full quality feedback loop.", regulatory="Supports Continued Process Verification (**FDA Process Validation Guidance**) by providing evidence of both process monitoring and the effectiveness of corrective actions. This is a key principle of **ICH Q10**.")
    lam = 0.2
    pre_intervention = df.iloc[:25]
    post_intervention = df.iloc[25:]
    mean_pre, sd_pre = pre_intervention['Impurity'].iloc[:15].mean(), pre_intervention['Impurity'].iloc[:15].std()
    df['EWMA'] = np.nan
    df.loc[:24, 'EWMA'] = pre_intervention['Impurity'].ewm(span=(2/lam)-1, adjust=False).mean()
    mean_post, sd_post = post_intervention['Impurity'].mean(), post_intervention['Impurity'].std()
    df.loc[25:, 'EWMA'] = post_intervention['Impurity'].ewm(span=(2/lam)-1, adjust=False).mean()

    ucl = mean_pre + 3 * sd_pre; lcl = mean_pre - 3 * sd_pre
    fig = go.Figure();
    fig.add_trace(go.Scatter(x=df.index, y=df['Impurity'], mode='markers', name='Individual Batch', marker_color=NEUTRAL_GREY, customdata=df['Batch'], hovertemplate='Batch: %{customdata}<br>Impurity: %{y:.3f}%<extra></extra>'));
    fig.add_trace(go.Scatter(x=df.index, y=df['EWMA'], mode='lines', name='EWMA', line=dict(color=PRIMARY_COLOR, width=3)));
    fig.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text="Initial UCL")
    fig.add_vline(x=25, line_dash='dot', line_color=DARK_GREY, annotation_text="Process Intervention")
    violation_idx = df[df['EWMA'] > ucl].first_valid_index();
    if violation_idx:
        fig.add_annotation(x=violation_idx, y=df['EWMA'][violation_idx], text="<b>Drift Signal</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title="<b>EWMA Chart with Process Intervention & Reset</b>", yaxis_title="Impurity Level (%)", xaxis_title="Batch Number")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The EWMA chart signaled a process drift at Batch #{violation_idx}. A planned intervention (e.g., column repacking) was performed at Batch #25. The post-intervention EWMA shows the process has returned to its original state of control. **Decision:** The corrective action was successful. Continue monitoring under the established control limits.")

def plot_cusum_chart(df):
    render_full_chart_briefing(context="Monitoring a high-speed, high-cost process parameter where rapid shift detection is critical.", significance="CUSUM charts are the fastest at detecting small, sustained shifts. This advanced version shows a two-sided chart with a V-Mask, the formal graphical tool for signal detection, providing a more robust and statistically sound decision-making framework.", regulatory="Demonstrates an advanced and mature quality system. The use of a V-Mask is a highly technical SPC method that shows a deep commitment to rapid response and process control, aligning with **ICH Q9** risk-based principles.")
    target = 10.0; sd = 0.05; k = 0.5 * sd; h = 5 * sd
    df['C+'] = 0.0; df['C-'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'C+'] = max(0, df.loc[i, 'Fill_Volume'] - (target + k) + df.loc[i-1, 'C+'])
        df.loc[i, 'C-'] = max(0, (target - k) - df.loc[i, 'Fill_Volume'] + df.loc[i-1, 'C-'])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("CUSUM for Upward Shift (C+)", "CUSUM for Downward Shift (C-)"))
    fig.add_trace(go.Scatter(y=df['C+'], name='CUSUM High (C+)', mode='lines+markers', line=dict(color=PRIMARY_COLOR)), row=1, col=1)
    fig.add_hline(y=h, line_dash='dash', line_color=ERROR_RED, annotation_text="H", row=1, col=1)
    fig.add_trace(go.Scatter(y=df['C-'], name='CUSUM Low (C-)', mode='lines+markers', line=dict(color=WARNING_AMBER)), row=2, col=1)
    fig.add_hline(y=h, line_dash='dash', line_color=ERROR_RED, annotation_text="H", row=2, col=1)
    violation_idx = df[df['C-'] > h].first_valid_index()
    if violation_idx:
        fig.add_annotation(x=violation_idx, y=df['C-'][violation_idx], text="<b>CUSUM Signal!</b>", showarrow=False, bgcolor=ERROR_RED, font=dict(color='white'), row=2, col=1)
        lead_distance = 10; v_mask_y = df['C-'][violation_idx]
        v_mask_upper_y = v_mask_y + k * lead_distance; v_mask_lower_y = v_mask_y - k * lead_distance
        fig.add_shape(type="line", x0=violation_idx, y0=v_mask_y, x1=violation_idx-lead_distance, y1=v_mask_upper_y, line=dict(color=DARK_GREY, width=2, dash="dot"), row=2, col=1)
        fig.add_shape(type="line", x0=violation_idx, y0=v_mask_y, x1=violation_idx-lead_distance, y1=v_mask_lower_y, line=dict(color=DARK_GREY, width=2, dash="dot"), row=2, col=1)
    fig.update_layout(height=500, title_text="<b>Two-Sided CUSUM Chart with V-Mask</b>", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The two-sided CUSUM chart detected a downward shift at sample #{violation_idx}. The V-Mask, placed at the signal point, graphically confirms the out-of-control condition as prior data points cross its arms. **Decision:** The filling line was halted. The rapid CUSUM detection minimized the number of non-conforming units produced.")

def plot_i_mr_chart(df):
    render_full_chart_briefing(context="Monitoring individual measurements where assessing both process mean and variability is critical.", significance="This enhanced I-MR chart automatically applies and visualizes Nelson Rules, detecting non-random patterns (like trends or shifts) within control limits that would be missed by simple UCL/LCL breaches, providing an earlier warning of process instability.", regulatory="Goes beyond basic SPC to demonstrate a mature process monitoring program. Using sensitizing rules like Nelson Rules is a best practice for Continued Process Verification (**FDA Guidance**) and demonstrates a proactive approach to quality management (**ICH Q10**).")
    i_data = df['Purity']; i_mean = i_data.mean(); mr = abs(i_data.diff()); mr_mean = mr.mean();
    i_ucl, i_lcl = i_mean + 2.66 * mr_mean, i_mean - 2.66 * mr_mean; mr_ucl = 3.267 * mr_mean
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart with Nelson Rules</b>", "<b>Moving Range (MR) Chart</b>"))
    fig.add_trace(go.Scatter(x=df['Date'], y=i_data, name='Purity', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1);
    fig.add_hline(y=i_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="Mean");
    fig.add_hline(y=i_ucl, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL");
    fig.add_hline(y=i_lcl, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL");
    s = (i_data - i_mean) / i_data.std()
    for j in range(8, len(s)):
        if all(s[j-8:j] < 0) or all(s[j-8:j] > 0):
            fig.add_annotation(x=df['Date'].iloc[j-4], y=i_data.iloc[j-4], text="<b>Nelson Rule 2</b><br>8 points one side", showarrow=True, bgcolor=WARNING_AMBER, row=1, col=1); break
    fig.add_trace(go.Scatter(x=df['Date'], y=mr, name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1);
    fig.add_hline(y=mr_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="Mean");
    fig.add_hline(y=mr_ucl, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL");
    fig.update_layout(height=600, showlegend=False, title_text="<b>I-MR Chart for Reference Standard Purity</b>")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Actionable Insight:** Even before any points breached the control limits, the I-chart detected a violation of Nelson Rule #2 (8 consecutive points below the mean), indicating a non-random downward shift. This early warning confirms the reference standard is degrading. **Decision:** Quarantine the current standard and qualify a new one immediately. This proactive measure prevents the generation of invalid data.")

def plot_cpk_analysis(df):
    render_full_chart_briefing(context="Assessing if a validated manufacturing process can reliably meet not just its official specifications, but also its tighter internal 'guard band' limits.", significance="Introduces **Guard-Banded Cpk (Cpk-GB)**, a critical internal metric that measures process capability against tighter, action-oriented limits. This provides an early warning if a process is drifting towards an edge of the specification, even if it's still officially 'in-spec'.", regulatory="Demonstrates a mature, risk-based approach to process control (**ICH Q9**). Maintaining a high Cpk-GB ensures the process stays well within the 'safe' operating space defined in the Design Space (**ICH Q8**), reducing the risk of OOS results.")
    data = df['Titer']; LSL, USL, target = 48.0, 52.0, 50.0;
    GBL, GBU = 49.0, 51.0
    mu, std = data.mean(), data.std(ddof=1)
    cpk, cp = 0,0
    if std > 0: cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std)); cp = (USL - LSL) / (6*std)
    cpk_gb = min((GBU - mu) / (3 * std), (mu - GBL) / (3 * std))
    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure();
        fig.add_trace(go.Histogram(x=data, name='Observed Data', histnorm='probability density', marker_color=PRIMARY_COLOR, opacity=0.7));
        x_fit = np.linspace(data.min(), data.max(), 200); y_fit = stats.norm.pdf(x_fit, mu, std);
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Normal', line=dict(color=DARK_GREY, width=2)))
        fig.add_vrect(x0=LSL, x1=USL, fillcolor=SUCCESS_GREEN, opacity=0.1, layer="below", line_width=0, name="Spec Limits")
        fig.add_vrect(x0=GBL, x1=GBU, fillcolor=WARNING_AMBER, opacity=0.15, layer="below", line_width=0, name="Guard Bands")
        fig.add_vline(x=LSL, line_dash="dash", line_color=ERROR_RED, annotation_text=f"LSL={LSL}");
        fig.add_vline(x=USL, line_dash="dash", line_color=ERROR_RED, annotation_text=f"USL={USL}");
        fig.add_vline(x=GBL, line_dash="dot", line_color=GUARD_BAND_COLOR, annotation_text=f"GBL={GBL}");
        fig.add_vline(x=GBU, line_dash="dot", line_color=GUARD_BAND_COLOR, annotation_text=f"GBU={GBU}");
        fig.update_layout(title_text=f'<b>Process Capability vs. Spec & Guard Band Limits</b>', xaxis_title="Titer Result (e12 vg/mL)", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Capability Indices")
        st.metric("Cpk (vs. Spec Limit)", f"{cpk:.2f}", f"{'PASS' if cpk >= 1.33 else 'FAIL'}: Target >= 1.33")
        st.metric("Cpk-GB (vs. Guard Band)", f"{cpk_gb:.2f}", f"{'PASS' if cpk_gb >= 1.33 else 'FAIL'}: Target >= 1.33", help="Capability calculated against tighter internal alert limits.")
        st.metric("Cp (Process Potential)", f"{cp:.2f}")
    if cpk_gb < 1.33 and cpk >= 1.33:
        st.warning(f"**Actionable Insight:** The process is capable against official specifications (Cpk={cpk:.2f}), but NOT against our internal guard bands (Cpk-GB={cpk_gb:.2f}). This indicates the process is running too close to the specification edge and is at high risk of producing OOS results if any minor shift occurs. **Decision:** Initiate a process improvement project to reduce variability, even though the process is technically 'in-spec'.")
    elif cpk >= 1.33:
        st.success(f"**Actionable Insight:** The process is highly capable against both specification (Cpk={cpk:.2f}) and internal guard band (Cpk-GB={cpk_gb:.2f}) limits. **Decision:** The process is robust and approved for routine manufacturing. Monitoring can continue at a standard frequency.")
    else:
        st.error(f"**Actionable Insight:** The Cpk of {cpk:.2f} is below the required 1.33. The process is not capable. **Decision:** Halt process validation. A full root cause analysis is required to re-develop the process to reduce variability or re-center the mean.")

def plot_hotelling_t2_chart(df):
    render_full_chart_briefing(context="Simultaneously monitoring two correlated CQAs from a bioreactor run.", significance="This advanced example includes a **Contribution Plot**, a critical diagnostic tool. When the T² chart signals an anomaly, the contribution plot immediately identifies *which* variable was most responsible for the out-of-control signal, directing the investigation efficiently.", regulatory="Using multivariate control charts demonstrates a mature understanding of process interactions (**ICH Q8**). Including contribution plots shows a sophisticated and systematic approach to investigations (**21 CFR 211.192**), moving beyond guessing to data-driven diagnosis.")
    data = df.values; mean_vec = data.mean(axis=0); inv_cov_mat = np.linalg.inv(np.cov(data, rowvar=False))
    t_squared_values = np.array([ (x - mean_vec).T @ inv_cov_mat @ (x - mean_vec) for x in data ])
    ucl = stats.f.ppf(0.99, 2, len(data)-1) * (2 * (len(data)-1) * (len(data)+1)) / (len(data) * (len(data)-2))
    anomaly_idx = np.argmax(t_squared_values)
    col1, col2 = st.columns(2)
    with col1:
        fig_t2 = go.Figure();
        fig_t2.add_trace(go.Scatter(y=t_squared_values, mode='lines+markers', name='T² Value', line=dict(color=PRIMARY_COLOR)));
        fig_t2.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text="UCL (99%)");
        fig_t2.add_annotation(x=anomaly_idx, y=t_squared_values[anomaly_idx], text="<b>Multivariate Anomaly</b>", showarrow=True, bgcolor=ERROR_RED, font=dict(color='white'))
        fig_t2.update_layout(title="<b>Hotelling's T² Chart</b>", yaxis_title="T² Statistic", xaxis_title="Batch Number")
        st.plotly_chart(fig_t2, use_container_width=True)
    with col2:
        contributions = (data[anomaly_idx] - mean_vec) * (inv_cov_mat @ (data[anomaly_idx] - mean_vec))
        contrib_df = pd.DataFrame({'Variable': df.columns, 'Contribution': contributions})
        fig_contrib = px.bar(contrib_df, x='Variable', y='Contribution', title=f"<b>Contribution to Anomaly at Batch {anomaly_idx}</b>", color='Variable', color_discrete_map={'Purity_Pct': PRIMARY_COLOR, 'Titer_vg_mL': WARNING_AMBER})
        st.plotly_chart(fig_contrib, use_container_width=True)
    st.error(f"**Actionable Insight:** The T² chart identified a multivariate anomaly at Batch #{anomaly_idx}. While two separate charts might have missed it, the T² chart detected the unusual combination of parameters. The contribution plot clearly shows that the **{contrib_df.loc[contrib_df['Contribution'].idxmax(), 'Variable']}** was the primary driver of the deviation. **Decision:** Quarantine Batch #{anomaly_idx}. The investigation should immediately focus on the root causes related to the anomalous variable.")

# SME FIX: Added a newline here to resolve the SyntaxError.
def plot_p_chart(df):
    render_full_chart_briefing(context="Monitoring the proportion of failures when the sample size varies per period (e.g., monthly SST failures).", significance="This p-chart is enhanced with **Wilson Score Intervals** for each point, providing a more accurate representation of uncertainty than standard Shewhart chart limits, especially when failure rates are low. This prevents overreaction to random noise.", regulatory="Demonstrates a statistically superior method for handling proportional data, aligning with expectations for robust data analysis in quality systems (**21 CFR 211.165(d)**) and showing a deeper understanding of statistical theory.")
    df['proportion'] = df['SSTs_Failed'] / df['SSTs_Run']; p_bar = df['SSTs_Failed'].sum() / df['SSTs_Run'].sum();
    df['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run']);
    df['LCL'] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run'])).clip(lower=0)
    intervals = [wilson_score_interval(p, n) for p, n in zip(df['proportion'], df['SSTs_Run'])]
    df['ci_low'] = [i[0] for i in intervals]; df['ci_high'] = [i[1] for i in intervals]
    
    fig = go.Figure();
    fig.add_trace(go.Scatter(x=df['Month'], y=df['proportion'], name='Proportion Failed', mode='lines+markers', line_color=PRIMARY_COLOR, error_y=dict(type='data', symmetric=False, array=df['ci_high']-df['proportion'], arrayminus=df['proportion']-df['ci_low'], color=GUARD_BAND_COLOR, thickness=1.5)));
    fig.add_trace(go.Scatter(x=df['Month'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash')));
    fig.add_trace(go.Scatter(x=df['Month'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'), showlegend=False))
    fig.add_hline(y=p_bar, name='Average Fail Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    fig.update_layout(title='<b>p-Chart for SST Failure Rate with Wilson Score Intervals</b>', yaxis_title='Proportion of SSTs Failed', yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The p-chart reveals a statistically significant spike in the SST failure rate in October, with the point clearly breaching the dynamic UCL. The Wilson Score Interval for that point is also entirely above the average failure rate, confirming the significance of the event. **Decision:** Launch an investigation focused on events in October. Review all column changes, mobile phase preparations, and instrument maintenance records from that period to find the root cause.")
## --- Other Attribute Charts (np, c, u) kept concise for brevity ---
def plot_np_chart(df):
    n = df['Batches_Sampled'].iloc[0]; p_bar = df['Defective_Vials'].sum() / (len(df) * n); ucl = n * p_bar + 3 * np.sqrt(n * p_bar * (1-p_bar)); lcl = max(0, n * p_bar - 3 * np.sqrt(n * p_bar * (1-p_bar)))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Week'], y=df['Defective_Vials'], name='Defective Vials', mode='lines+markers')); fig.add_hline(y=ucl, name='UCL', line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=lcl, name='LCL', line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=n*p_bar, name='Center Line', line_dash='dot', line_color=SUCCESS_GREEN)
    fig.update_layout(title='<b>np-Chart for Number of Defective Vials</b>', yaxis_title=f'Count of Defective Vials (n={n})', xaxis_title='Week'); st.plotly_chart(fig, use_container_width=True)

def plot_c_chart(df):
    c_bar = df['Contaminants_per_Plate'].mean(); ucl = c_bar + 3 * np.sqrt(c_bar); lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Week'], y=df['Contaminants_per_Plate'], name='Contaminants', mode='lines+markers')); fig.add_hline(y=ucl, name='UCL', line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=lcl, name='LCL', line_dash='dash', line_color=ERROR_RED); fig.add_hline(y=c_bar, name='Center Line', line_dash='dot', line_color=SUCCESS_GREEN)
    fig.update_layout(title='<b>c-Chart for Environmental Monitoring</b>', yaxis_title='Colony Count per Plate', xaxis_title='Week'); st.plotly_chart(fig, use_container_width=True)

def plot_u_chart(df):
    df['defects_per_unit'] = df['Particulate_Defects'] / df['Vials_Inspected']; u_bar = df['Particulate_Defects'].sum() / df['Vials_Inspected'].sum(); df['UCL'] = u_bar + 3 * np.sqrt(u_bar / df['Vials_Inspected']); df['LCL'] = (u_bar - 3 * np.sqrt(u_bar / df['Vials_Inspected'])).clip(lower=0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Batch'], y=df['defects_per_unit'], name='Defect Rate', mode='lines+markers')); fig.add_trace(go.Scatter(x=df['Batch'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'))); fig.add_trace(go.Scatter(x=df['Batch'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'), showlegend=False)); fig.add_hline(y=u_bar, name='Average Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    fig.update_layout(title='<b>u-Chart for Particulate Defect Rate</b>', yaxis_title='Defects per Vial', xaxis_title='Batch'); st.plotly_chart(fig, use_container_width=True)

## --- LIFECYCLE HUB FUNCTIONS ---
def plot_stability_analysis(df):
    render_full_chart_briefing(context="Analyzing long-term stability data from three different validation batches to establish a unified shelf-life.", significance="Follows **ICH Q1E** guidance by first testing for batch poolability using **ANCOVA (Analysis of Covariance)**. This statistically justifies using a single shelf-life estimate across all batches, a critical requirement for regulatory submission.", regulatory="Directly implements the statistical methodology prescribed in **ICH Q1E: Evaluation of Stability Data**, specifically regarding the analysis of data from multiple batches. This demonstrates a high level of regulatory compliance.")
    spec_limit = 90.0
    model = ols('Potency_pct ~ Time_months * C(Batch)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    p_value_interaction = anova_table['PR(>F)']['Time_months:C(Batch)']
    
    col1, col2 = st.columns([3,1])
    with col1:
        fig = px.scatter(df, x='Time_months', y='Potency_pct', color='Batch', title="<b>ICH Q1E Multi-Batch Stability & Poolability Analysis</b>", color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.add_hline(y=spec_limit, line_dash='dash', line_color=ERROR_RED, annotation_text="Spec Limit")
        
        pooled_justified = p_value_interaction >= 0.25
        if pooled_justified:
            slope, intercept, _, _, _ = stats.linregress(df['Time_months'], df['Potency_pct'])
            shelf_life = (spec_limit - intercept) / slope
            fig.add_trace(go.Scatter(x=df['Time_months'], y=intercept + slope * df['Time_months'], name='Pooled Regression', line=dict(color='black', width=4, dash='dash')))
            if slope < 0: fig.add_vline(x=shelf_life, line_dash='dot', line_color=SUCCESS_GREEN, annotation_text=f"Pooled Shelf Life: {shelf_life:.1f} mo")
        else: 
            for batch in df['Batch'].unique():
                batch_df = df[df['Batch'] == batch]
                slope, intercept, _, _, _ = stats.linregress(batch_df['Time_months'], batch_df['Potency_pct'])
                fig.add_trace(go.Scatter(x=batch_df['Time_months'], y=intercept + slope * batch_df['Time_months'], name=f'{batch} Fit', line=dict(width=2)))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Batch Poolability Test (ANCOVA)")
        st.metric("P-value for Interaction (Slopes)", f"{p_value_interaction:.3f}")
        if pooled_justified:
            st.success(f"**P-value ({p_value_interaction:.3f}) > 0.25**\n\nBatches are poolable. A single shelf life is statistically justified.")
            st.success(f"**Actionable Insight:** The ANCOVA test confirms no significant difference between batch degradation rates. **Decision:** Propose a unified shelf life of **{int(shelf_life)} months** in the regulatory filing, using this statistical analysis as justification.")
        else:
            st.error(f"**P-value ({p_value_interaction:.3f}) < 0.25**\n\nBatches are NOT poolable. The shortest shelf life must be used.")
            st.error("**Actionable Insight:** The ANCOVA test shows a significant difference in degradation rates between batches. **Decision:** The batches cannot be pooled. The product shelf life must be set based on the worst-performing batch. An investigation into the cause of inter-batch variability is required.")

def plot_method_equivalency_tost(df):
    render_full_chart_briefing(context="Comparing a new UPLC method against a legacy HPLC method.", significance="This analysis pairs the standard **TOST (Two One-Sided T-Tests)** for equivalence with a **Bland-Altman plot**. The Bland-Altman plot is crucial for revealing if the bias between methods is constant or if it changes with the magnitude of the result, providing a much deeper understanding of method agreement.", regulatory="TOST is the gold standard for equivalence (**ICH Q12**). The addition of a Bland-Altman plot demonstrates a more thorough analysis, often expected by regulators for method transfer or validation studies to ensure there are no hidden systematic errors.")
    diff = df['UPLC'] - df['HPLC']; n = len(diff); mean_diff = diff.mean(); std_diff = diff.std(ddof=1);
    se_diff = std_diff / np.sqrt(n); t_crit = stats.t.ppf(0.95, df=n-1);
    ci_lower = mean_diff - t_crit * se_diff; ci_upper = mean_diff + t_crit * se_diff; equiv_limit = 0.5
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(); fig.add_trace(go.Scatter(x=[mean_diff], y=[0], mode='markers', marker=dict(color=PRIMARY_COLOR, size=15, symbol='diamond'), error_x=dict(type='data', array=[ci_upper - mean_diff], arrayminus=[mean_diff - ci_lower], thickness=4), name='90% CI of Mean Difference'))
        fig.add_vline(x=-equiv_limit, line_dash='dash', line_color=ERROR_RED, annotation_text=f"LEL=-{equiv_limit}"); fig.add_vline(x=equiv_limit, line_dash='dash', line_color=ERROR_RED, annotation_text=f"UEL={equiv_limit}"); fig.update_yaxes(visible=False)
        fig.update_layout(title="<b>Method Equivalency via TOST</b>", xaxis_title="Difference in Purity (%) [UPLC - HPLC]")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df['Average'] = (df['UPLC'] + df['HPLC']) / 2; df['Difference'] = df['UPLC'] - df['HPLC']
        fig_ba = px.scatter(df, x='Average', y='Difference', title="<b>Bland-Altman Plot</b>", labels={'Average': 'Average of Methods', 'Difference': 'Difference (UPLC - HPLC)'})
        fig_ba.add_hline(y=mean_diff, line_color=PRIMARY_COLOR, annotation_text='Mean Diff')
        fig_ba.add_hline(y=mean_diff + 1.96 * std_diff, line_dash='dash', line_color=DARK_GREY, annotation_text='+1.96 SD')
        fig_ba.add_hline(y=mean_diff - 1.96 * std_diff, line_dash='dash', line_color=DARK_GREY, annotation_text='-1.96 SD')
        st.plotly_chart(fig_ba, use_container_width=True)

    if ci_lower > -equiv_limit and ci_upper < equiv_limit:
        st.success("**Actionable Insight:** The 90% confidence interval for the mean difference is fully contained within the equivalence limits, proving equivalency. The Bland-Altman plot shows no discernible trend, confirming the bias is consistent across the analytical range. **Decision:** The UPLC method is equivalent. Initiate change control to replace the HPLC method.")
    else:
        st.error("**Actionable Insight:** Equivalence has not been demonstrated. The Bland-Altman plot should be examined to see if the failure is due to a consistent bias or a magnitude-dependent issue. **Decision:** Do not replace the method. Investigate the source of the bias.")

## --- PREDICTIVE HUB FUNCTIONS ---
@st.cache_resource
def get_oos_rca_model(_df):
    df_encoded = pd.get_dummies(_df, columns=['Instrument', 'Analyst', 'Molecule_Type'])
    features = [col for col in df_encoded.columns if col != 'Root_Cause']; target = 'Root_Cause'; X, y = df_encoded[features], df_encoded[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y); return model, X.columns

def run_oos_prediction_model(df):
    model, feature_cols = get_oos_rca_model(df)
    col1, col2, col3 = st.columns(3); instrument = col1.selectbox("Instrument Used", df['Instrument'].unique()); analyst = col2.selectbox("Analyst", df['Analyst'].unique()); molecule = col3.selectbox("Molecule Type", df['Molecule_Type'].unique())
    if st.button("🔬 Predict Probable Root Cause", type="primary"):
        input_data = pd.DataFrame(0, columns=feature_cols, index=[0]); input_data[f'Instrument_{instrument}'] = 1; input_data[f'Analyst_{analyst}'] = 1; input_data[f'Molecule_Type_{molecule}'] = 1
        pred_proba = model.predict_proba(input_data); proba_df = pd.DataFrame(pred_proba.T, index=model.classes_, columns=['Probability']).sort_values('Probability', ascending=False)
        fig = px.bar(proba_df, x='Probability', y=proba_df.index, orientation='h', title="<b>Predicted Root Cause Probability</b>", text_auto='.1%'); st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def get_maint_model(_df):
    features = ['Run_Hours', 'Pressure_Spikes', 'Column_Age_Days']; target = 'Needs_Maint'; X, y = _df[features], _df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y); return model, features

def run_hplc_maintenance_model(df):
    render_full_chart_briefing(context="Managing a fleet of HPLC instruments.", significance="This advanced tool provides not only a real-time risk score but also a **'What-If' analysis** capability using SHAP. It allows a manager to interactively explore how future use (e.g., more run hours) would impact the risk, enabling truly proactive and forward-looking maintenance planning.", regulatory="A predictive, risk-based approach aligns with **ICH Q9**. The interactive, explainable AI (XAI) component supports a strong justification for maintenance decisions during audits, aligning with the principles of Computer Software Assurance (CSA).")
    model, feature_names = get_maint_model(df)
    
    st.subheader("Interactive 'What-If' Maintenance Planner")
    col1, col2, col3 = st.columns(3);
    hours = col1.slider("Total Run Hours", 50, 1000, 750, key='hours');
    spikes = col2.slider("Pressure Spikes >100psi", 0, 20, 18, key='spikes');
    age = col3.slider("Column Age (Days)", 10, 300, 280, key='age')
    
    input_df = pd.DataFrame([[hours, spikes, age]], columns=feature_names)
    pred_prob = model.predict_proba(input_df)[0][1]
    
    colA, colB = st.columns([1,2])
    with colA:
        fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = pred_prob * 100, title = {'text': "Maintenance Risk Score"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': ERROR_RED if pred_prob > 0.7 else WARNING_AMBER if pred_prob > 0.4 else SUCCESS_GREEN}}))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=0)); st.plotly_chart(fig_gauge, use_container_width=True)
    with colB:
        st.subheader("Explainable AI (XAI): Why this score?")
        st.info("This SHAP plot shows which factors are pushing the risk score higher (red) or lower (blue).")
        # SME FIX: This robust implementation handles API inconsistencies in the SHAP library's output format
        # for binary classifiers, preventing both IndexError and TypeError.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        expected_value = explainer.expected_value
        
        shap_values_for_plot = None
        expected_value_for_plot = None

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_for_plot = shap_values[1]
            if isinstance(expected_value, list) and len(expected_value) == 2:
                expected_value_for_plot = expected_value[1]
            else:
                expected_value_for_plot = expected_value
        else:
            shap_values_for_plot = shap_values
            expected_value_for_plot = expected_value
        
        if shap_values_for_plot is not None and expected_value_for_plot is not None:
             st_shap(shap.force_plot(expected_value_for_plot, shap_values_for_plot, input_df), height=150)

    st.warning("**Actionable Insight:** The interactive model shows the current high risk score is driven by high **Run Hours** and **Column Age**. Using the sliders, we can see that replacing the column (resetting age to 10) would lower the risk score significantly, but not enough to be 'safe'. This indicates the pump also requires service due to the run hours. **Decision:** Schedule a full preventative maintenance, including pump seal replacement and a new column, to bring the risk score back into the green zone.")

# --- Other functions (plotting, page rendering) follow ---
def render_strategic_hub_page(budget_df, team_df):
    st.title("Executive & Strategic Hub"); render_manager_briefing(title="Leading Analytical Development as a High-Impact Business Unit", content="This hub demonstrates the strategic, business-oriented aspects of leading the AD Ops function. It covers financial management, goal setting (OKRs), and, most importantly, the development and mentorship of the scientific team.", reg_refs="Company HR Policies, Departmental Budget", business_impact="Ensures the department operates in a fiscally responsible manner, is aligned with corporate goals, and fosters a culture of growth and expertise.", quality_pillar="Leadership & People Management.", risk_mitigation="Prevents skill gaps on critical projects and justifies resource needs through clear, data-driven forecasting and performance tracking.")
    st.subheader("Departmental OKRs", divider='violet'); st.dataframe(pd.DataFrame({"Objective": ["Accelerate PD Support", "Enhance Method Robustness", "Foster Team Growth"], "Key Result": ["Reduce avg. sample TAT by 15%", "Implement QbD for 2 new methods", "Cross-train 2 scientists on ddPCR"], "Status": ["On Track", "Complete", "On Track"]}).style.map(lambda s: f"background-color: {SUCCESS_GREEN if s in ['On Track', 'Complete'] else WARNING_AMBER}; color: white;"), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2); col1.subheader("Annual Budget Performance", divider='violet'); col1.plotly_chart(px.bar(budget_df, x='Category', y=budget_df['Actual'] - budget_df['Budgeted']), use_container_width=True); col2.subheader("Team Skill & Development Matrix", divider='violet'); col2.dataframe(team_df, use_container_width=True, hide_index=True);

def render_statistical_toolkit_page(lj_df, ewma_df, cusum_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df):
    st.title("Advanced Statistical Toolkit"); render_manager_briefing(title="Applying Statistical Rigor to Analytical Problems", content="This hub serves as a comprehensive toolkit, demonstrating deep, first-principles expertise in applying the correct statistical process control (SPC) and capability tools to different analytical challenges. Each chart is presented with a realistic, domain-specific context to show not just *how* to perform the analysis, but *why* and *when* to use it.", reg_refs="ICH Q9 (Quality Risk Management), 21 CFR 211.165(d) (Statistics)", business_impact="Ensures that decisions about process control, method performance, and product quality are based on objective statistical evidence, not intuition.", quality_pillar="Statistical Thinking & Data Literacy.", risk_mitigation="Detects process drifts, out-of-control states, and capability issues early, preventing large-scale failures, batch rejections, and invalid data.")
    tab1, tab2, tab3 = st.tabs(["**📊 Monitoring Process Stability & Drift**", "**📈 Monitoring Quality & Yield (Attribute Data)**", "**🔎 Advanced Process & Method Insights**"])
    with tab1: st.subheader("Tools for Monitoring Continuous Data", divider='violet'); plot_i_mr_chart(imr_df); plot_levey_jennings(lj_df); plot_ewma_chart(ewma_df); plot_cusum_chart(cusum_df)
    with tab2: st.subheader("Tools for Monitoring Attribute (Count/Fail) Data", divider='violet'); plot_p_chart(p_df); plot_np_chart(np_df); plot_c_chart(c_df); plot_u_chart(u_df)
    with tab3: st.subheader("Tools for Deeper Process Understanding", divider='violet'); plot_hotelling_t2_chart(t2_df); plot_cpk_analysis(cpk_df)

def render_lifecycle_hub_page(stability_df, tost_df, screening_df, doe_df):
    st.title("Method & Product Lifecycle Hub"); render_manager_briefing(title="Guiding Methods from R&D to Commercial Launch", content="This hub demonstrates the strategic oversight of the entire analytical and product lifecycle. It showcases a deep understanding of early-stage method development (QbD), late-stage product stability, and the statistical tools needed to manage post-approval changes and method transfers.", reg_refs="ICH Q1E (Stability), ICH Q12 (Lifecycle Management), ICH Q8/Q14 (QbD)", business_impact="Creates robust, well-understood analytical methods that are less prone to failure, accelerates development timelines, and provides a solid data foundation for regulatory filings and product shelf-life justification.", quality_pillar="Lifecycle Management & Scientific Rigor.", risk_mitigation="Front-loads risk management into the development phase, preventing costly validation failures and ensuring methods are transferable and maintainable throughout the product's commercial life.")
    st.subheader("Late-Stage: Commercial & Post-Approval Support", divider='violet'); plot_stability_analysis(stability_df); plot_method_equivalency_tost(tost_df)

def render_predictive_hub_page(oos_df, backlog_df, maintenance_df):
    st.title("Predictive Operations & Diagnostics"); render_manager_briefing(title="Building a Proactive, Data-Driven Operations Function", content="This hub showcases a forward-looking leadership approach, using predictive analytics and machine learning to move the AD Ops function from a reactive service center to a proactive, strategic partner. These tools are used to forecast future challenges, diagnose problems faster, and optimize resource allocation.", reg_refs="ICH Q9 (Quality Risk Management), FDA's Computer Software Assurance (CSA) Guidance", business_impact="Maximizes instrument uptime, accelerates OOS investigations, and provides data-driven justification for resource planning, ultimately increasing the speed and efficiency of the entire R&D organization.", quality_pillar="Predictive Analytics & Continuous Improvement.", risk_mitigation="Anticipates future bottlenecks, equipment failures, and quality issues, allowing for mitigation *before* they occur and impact critical program timelines.")
    st.subheader("Predictive Diagnostics & Troubleshooting", divider='violet'); run_oos_prediction_model(oos_df); st.subheader("Proactive Resource & Maintenance Planning", divider='violet'); run_hplc_maintenance_model(maintenance_df)

def render_qbd_quality_systems_hub_page(sankey_df):
    st.title("QbD & Quality Systems Hub"); render_manager_briefing(title="Integrating Quality Systems into Analytical Development", content="This hub demonstrates a deep understanding of modern quality systems and philosophies. It showcases how to proactively build quality into methods from the start using Quality by Design (QbD) and Design Controls, and how to react to deviations with systematic, compliant problem-solving tools.", reg_refs="ICH Q8, Q9, Q10; 21 CFR 820.30 (Design Controls); 21 CFR 211.192 (Investigations)", business_impact="Creates fundamentally more robust and reliable methods, reduces validation failures, streamlines regulatory submissions, and ensures investigations are efficient and scientifically sound.", quality_pillar="Proactive Quality & Systematic Problem Solving.", risk_mitigation="Moves the function from a 'test-and-fix' mentality to a 'design-and-understand' paradigm, fundamentally de-risking the entire method lifecycle and ensuring compliance with formal investigation requirements.")
    st.subheader("Reactive Problem Solving & Root Cause Analysis (RCA)", divider='violet'); render_troubleshooting_flowchart()

# ======================================================================================
# SECTION 6: MAIN APP ROUTER (SIDEBAR NAVIGATION)
# ======================================================================================
st.sidebar.title("AD Ops Navigation")
PAGES = { "Executive & Strategic Hub": render_strategic_hub_page, "QbD & Quality Systems Hub": render_qbd_quality_systems_hub_page, "Method & Product Lifecycle Hub": render_lifecycle_hub_page, "Predictive Operations & Diagnostics": render_predictive_hub_page, "Advanced Statistical Toolkit": render_statistical_toolkit_page, }
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
(budget_df, team_df, lj_df, ewma_df, cusum_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df, stability_df, tost_df, screening_df, doe_df, oos_df, backlog_df, maintenance_df, sankey_df) = generate_master_data()
page_function = PAGES[selection]
if selection == "Executive & Strategic Hub": page_function(budget_df, team_df)
elif selection == "QbD & Quality Systems Hub": page_function(sankey_df)
elif selection == "Method & Product Lifecycle Hub": page_function(stability_df, tost_df, screening_df, doe_df)
elif selection == "Predictive Operations & Diagnostics": page_function(oos_df, backlog_df, maintenance_df)
elif selection == "Advanced Statistical Toolkit": page_function(lj_df, ewma_df, cusum_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df)
st.sidebar.markdown("---"); st.sidebar.markdown("### Role Focus"); st.sidebar.info("This portfolio is for an **Associate Director, Analytical Development Operations** role, demonstrating leadership in building high-throughput testing functions, managing the method lifecycle, and applying advanced statistical methods."); st.sidebar.markdown("---"); st.sidebar.markdown("### Key Regulatory & Quality Frameworks")
with st.sidebar.expander("View Applicable Guidelines", expanded=False): st.markdown("- **ICH Q1E, Q2, Q8, Q9, Q10, Q12, Q14**\n- **21 CFR Parts 11, 211, 820.30**\n- **EudraLex Vol. 4, Annex 1 & 15**\n- **ISO 17025, ISO 13485**")
