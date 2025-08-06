# ======================================================================================
# ANALYTICAL DEVELOPMENT OPERATIONS COMMAND CENTER
# v13.6 - Final, Complete with Technical Deep Dive
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
# SME Note: The SHAP library has been removed due to instability and replaced with a more robust alternative.

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
    [data-testid="stDataFrame"] {{
        width: 100%;
    }}
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
    # --- Data for EXECUTIVE & STRATEGIC HUB ---
    team_df = pd.DataFrame({
        'Scientist': ['J. Doe (Lead)', 'S. Smith', 'M. Lee', 'K. Chen'], 
        'Role': ['Assoc. Director', 'Sr. Scientist', 'Scientist II', 'Scientist I'], 
        'Expertise': ['HPLC/CE', 'ddPCR/qPCR', 'Cell-Based Assays', 'ELISA'],
        'Strategic Need': ['Program Leadership', 'AAV-101 Program', 'High-throughput Screening', 'Routine Testing Support'],
        'Development Plan': ['Mentor team on QbD', 'Lead AAV-101 analytics', 'Cross-train on ddPCR & automation', 'Gain validation experience']
    })
    tat_data = pd.DataFrame({
        'Month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01']),
        'TAT_Days': [12.5, 11.0, 10.2, 9.5, 8.1, 7.5]
    })
    program_data = pd.DataFrame({
        'Program': ['AAV-101', 'mAb-202', 'mRNA-301'],
        'Phase': ['Phase 1', 'Pre-clinical', 'Research'],
        'Analytical Lead': ['S. Smith', 'J. Doe', 'M. Lee'],
        'Key Upcoming Milestone': ['Phase 1 Potency Matrix Validation', 'Candidate Selection Assay Panel', 'Initial Titer Method Dev.'],
        'Status': ['On Track', 'Minor Delays', 'At Risk']
    })
    tech_roadmap_data = pd.DataFrame([
        {'Quarter': 'Q2', 'Initiative': 'Automated Liquid Handler', 'Impact': 'Reduce sample prep time by 75%', 'Projected FTE Savings (hrs/wk)': 20, 'Est. Capital Cost ($K)': 150, 'Projected ROI (Months)': 18, 'Status': '✅ Deployed'},
        {'Quarter': 'Q3', 'Initiative': 'LIMS Integration', 'Impact': 'Eliminate manual data transcription errors', 'Projected FTE Savings (hrs/wk)': 10, 'Est. Capital Cost ($K)': 75, 'Projected ROI (Months)': 24, 'Status': 'In Progress'},
        {'Quarter': 'Q4', 'Initiative': 'Next-Gen ddPCR', 'Impact': 'Increase sample throughput by 2x', 'Projected FTE Savings (hrs/wk)': 15, 'Est. Capital Cost ($K)': 250, 'Projected ROI (Months)': 30, 'Status': 'Evaluation'}
    ])
    workflow_data = pd.DataFrame({
        'Stage': ["1. Sample Received", "2. Sample Prep", "3. Instrument Run", "4. Data Analysis", "5. Data Review/Approval", "6. Report Issued"],
        'Samples': [250, 248, 245, 245, 180, 175]
    })
    training_data = pd.DataFrame({
        'Module': ['SOP-001: General Lab Safety', 'SOP-102: HPLC Operation', 'SOP-205: ddPCR Data Analysis', 'Annual GxP Refresher'],
        'Team Completion': [100, 75, 50, 100] # Stored as percentages
    })
    program_analytical_methods = {
        'AAV-101': pd.DataFrame({
            'Method': ['Titer (ddPCR)', 'Purity (CE-SDS)', 'Potency (Cell-Based)'],
            'Status': ['Transferred', 'Validating', 'In Development'],
            'Key Performance Characteristic': ['Precision <15% RSD', 'LOQ < 0.5%', 'S/N > 10']
        }),
        'mAb-202': pd.DataFrame({
            'Method': ['Titer (Protein A)', 'Glycan Profile (HILIC)', 'Charge Variants (iCIEF)'],
            'Status': ['Transferred', 'Transferred', 'Development At Risk'],
            'Key Performance Characteristic': ['Linearity > 0.99', 'Peak Resolution > 2.0', 'TBD']
        }),
        'mRNA-301': pd.DataFrame({
            'Method': ['Concentration (UV)', 'Capping Efficiency (HPLC)', 'Poly(A) Tail (LC-MS)'],
            'Status': ['In Development', 'In Development', 'Research'],
            'Key Performance Characteristic': ['TBD', 'TBD', 'TBD']
        })
    }
    risk_register_data = pd.DataFrame([
        {'Program': 'AAV-101', 'Partner Team': 'Downstream', 'Dependency': 'Purity method for new column resin', 'Risk Level': 'High', 'Mitigation Plan': 'Bridge study initiated, completion EOW.'},
        {'Program': 'mAb-202', 'Partner Team': 'Formulation', 'Dependency': 'Sub-visible particle method for high-concentration formula', 'Risk Level': 'Medium', 'Mitigation Plan': 'Evaluating two orthogonal techniques in parallel.'},
        {'Program': 'AAV-101', 'Partner Team': 'Upstream', 'Dependency': 'Rapid in-process titer result', 'Risk Level': 'Low', 'Mitigation Plan': 'Existing ddPCR method is sufficient.'}
    ])
    
    # --- Data for LIFECYCLE HUB ---
    transfer_data = pd.DataFrame([
        {'Program': 'AAV-101', 'Method': 'Potency Assay', 'Task': 'Draft Transfer Protocol', 'Status': '✅ Done'},
        {'Program': 'AAV-101', 'Method': 'Potency Assay', 'Task': 'Train QC Analysts', 'Status': 'In Progress'},
        {'Program': 'AAV-101', 'Method': 'Potency Assay', 'Task': 'Provide Qualified Reagents', 'Status': 'In Progress'},
        {'Program': 'AAV-101', 'Method': 'Potency Assay', 'Task': 'Execute Protocol', 'Status': 'Not Started'},
        {'Program': 'mAb-202', 'Method': 'Charge Variant Assay', 'Task': 'Draft Transfer Protocol', 'Status': 'At Risk'},
        {'Program': 'mAb-202', 'Method': 'Charge Variant Assay', 'Task': 'Train QC Analysts', 'Status': 'Not Started'},
    ])
    validation_data = pd.DataFrame({
        'Method': ['AAV Titer (ddPCR)', 'AAV Titer (ddPCR)', 'AAV Titer (ddPCR)', 'mAb Purity (HPLC)', 'mAb Purity (HPLC)', 'mAb Purity (HPLC)'],
        'Parameter': ['Accuracy', 'Precision (Inter)', 'Linearity (R²)', 'Accuracy', 'Precision (Inter)', 'Linearity (R²)'],
        'Result': [98.5, 4.5, 0.9992, 101.2, 1.8, 0.9998],
        'Acceptance Criteria': ['95-105%', '< 15% RSD', '> 0.995', '98-102%', '< 2.0% RSD', '> 0.999'],
        'Status': ['Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']
    })
    
    # --- Data for STATISTICAL TOOLKIT ---
    lj_data = np.random.normal(100.0, 2.0, 30); lj_data[20] = 106.5; lj_data[25:27] = [104.5, 104.8]; lj_df = pd.DataFrame({'Value': lj_data, 'Analyst': np.random.choice(['Smith', 'Lee'], 30)})
    ewma_data_pre = np.random.normal(0.5, 0.05, 25); ewma_data_post = np.random.normal(0.5, 0.05, 15); ewma_df = pd.DataFrame({'Impurity': np.concatenate([ewma_data_pre, ewma_data_post]), 'Batch': [f"AAV101-B{100+i}" for i in range(40)]}); ewma_df.loc[15:24, 'Impurity'] += 0.04
    cusum_data = np.random.normal(10.0, 0.05, 50); cusum_data[25:] -= 0.04; cusum_df = pd.DataFrame({'Fill_Volume': cusum_data, 'Nozzle': np.random.choice([1,2,3,4], 50)})
    zone_data = np.random.normal(20, 0.5, 25); zone_data[15:] -= 0.4; zone_df = pd.DataFrame({'Seal_Strength': zone_data, 'Operator': np.random.choice(['Op-A', 'Op-B'], 25)})
    imr_data = np.random.normal(99.5, 0.1, 100); imr_data[60:68] -= 0.2; imr_data[80:] -= 0.25; imr_df = pd.DataFrame({'Purity': imr_data, 'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))})
    cpk_data = np.random.normal(50.5, 0.25, 150); cpk_df = pd.DataFrame({'Titer': cpk_data})
    mean_vec = [95, 10.0]; std_devs = [1.5, 1.5]; correlation = 0.7; cov_mat = [[std_devs[0]**2, correlation * std_devs[0] * std_devs[1]], [correlation * std_devs[0] * std_devs[1], std_devs[1]**2]]
    t2_data_in = np.random.multivariate_normal(mean_vec, np.array(cov_mat), 30)
    t2_outlier = [97.5, 7.5]; t2_data = np.vstack([t2_data_in[:24], t2_outlier, t2_data_in[24:]]); t2_df = pd.DataFrame(t2_data, columns=['Purity_Pct', 'Titer (e12 vg/mL)'])
    p_data = {'Month': pd.to_datetime(pd.date_range(start='2023-01-01', periods=12, freq='ME')), 'SSTs_Run': np.random.randint(40, 60, 12)}; p_df = pd.DataFrame(p_data); p_df['SSTs_Failed'] = np.random.binomial(n=p_df['SSTs_Run'], p=0.05); p_df.loc[9, 'SSTs_Failed'] = 8
    np_df = pd.DataFrame({'Week': range(1, 21), 'Batches_Sampled': 50, 'Defective_Vials': np.random.binomial(n=50, p=0.04, size=20)}); np_df.loc[12, 'Defective_Vials'] = 7
    c_df = pd.DataFrame({'Week': range(1, 21), 'Contaminants_per_Plate': np.random.poisson(lam=3, size=20)}); c_df.loc[15, 'Contaminants_per_Plate'] = 9
    u_data = {'Batch': range(1, 16), 'Vials_Inspected': np.random.randint(50, 150, 15)}; u_df = pd.DataFrame(u_data); u_df['Particulate_Defects'] = np.random.poisson(lam=u_df['Vials_Inspected'] * 0.02); u_df.loc[10, 'Particulate_Defects'] = 8
    time_pts = [0, 3, 6, 9, 12, 18, 24]; stability_dfs = []
    for i in range(3):
        base_potency = 101.2 - i*0.5; slope = -0.2 - i*0.02 + np.random.normal(0, 0.01)
        potency = base_potency + slope * np.array(time_pts) + np.random.normal(0, 0.2, len(time_pts))
        stability_dfs.append(pd.DataFrame({'Batch': f'Batch {i+1}', 'Time_months': time_pts, 'Potency_pct': potency}))
    stability_df = pd.concat(stability_dfs, ignore_index=True)
    tost_df = pd.DataFrame({'HPLC': np.random.normal(98.5, 0.5, 30), 'UPLC': np.random.normal(98.7, 0.4, 30)})
    screening_df = pd.DataFrame({'Factor': ['Temp', 'pH', 'Flow_Rate', 'Gradient', 'Column_Lot', 'Analyst'], 'Effect_Size': [0.2, 1.8, 0.1, 1.5, 0.3, 0.2]})
    doe_df = pd.DataFrame(np.random.uniform(-1, 1, (15, 2)), columns=['pH', 'Gradient_Slope']); doe_df['Peak_Resolution'] = 2.5 - 0.5*doe_df['pH']**2 - 0.8*doe_df['Gradient_Slope']**2 + 0.3*doe_df['pH']*doe_df['Gradient_Slope'] + np.random.normal(0, 0.1, 15)
    oos_df = pd.DataFrame({'Instrument': np.random.choice(['HPLC-01', 'HPLC-02', 'CE-01'], 100), 'Analyst': np.random.choice(['Smith', 'Lee', 'Chen'], 100), 'Molecule_Type': np.random.choice(['mAb', 'AAV'], 100), 'Root_Cause': np.random.choice(['Sample_Prep_Error', 'Instrument_Malfunction', 'Column_Issue'], 100, p=[0.5, 0.3, 0.2])})
    backlog_vals = 10 + np.arange(104)*0.5 + np.random.normal(0, 5, 104) + np.sin(np.arange(104)/8)*5
    backlog_df = pd.DataFrame({'Week': pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W')), 'Backlog': backlog_vals.clip(min=0)})
    maintenance_df = pd.DataFrame({'Run_Hours': np.random.randint(50, 1000, 100), 'Pressure_Spikes': np.random.randint(0, 20, 100), 'Column_Age_Days': np.random.randint(10, 300, 100)}); maintenance_df['Needs_Maint'] = (maintenance_df['Run_Hours'] > 600) | (maintenance_df['Pressure_Spikes'] > 15) | (maintenance_df['Column_Age_Days'] > 250)
    sankey_df = pd.DataFrame({'Source': ['Column Lot', 'Mobile Phase Purity', 'Gradient Slope', 'Flow Rate', 'Column Temp', 'Peak Resolution', 'Assay Accuracy'], 'Target': ['Peak Resolution', 'Peak Resolution', 'Peak Resolution', 'Assay Precision', 'Assay Accuracy', 'Final Purity Result', 'Final Purity Result'], 'Value': [8, 5, 10, 7, 6, 12, 10]})
    automation_candidates = pd.DataFrame({
        'Assay': ['ELISA (mAb-202)', 'ddPCR Plate Prep', 'CE-SDS Sample Prep', 'HPLC Mobile Phase Prep'],
        'Manual Throughput (Samples/FTE/Wk)': [40, 80, 60, 200],
        'Est. Automation Throughput': [400, 400, 180, 200],
        'Error Rate (Manual)': [0.08, 0.05, 0.03, 0.01],
        'Est. Error Rate (Auto)': [0.01, 0.01, 0.01, 0.01],
        'Automation Priority Score': [9.5, 8.0, 6.5, 2.0]
    })

    return (team_df, lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, 
            p_df, np_df, c_df, u_df, stability_df, tost_df, screening_df, doe_df, 
            oos_df, backlog_df, maintenance_df, sankey_df, 
            tat_data, program_data, tech_roadmap_data, workflow_data, training_data, 
            program_analytical_methods, transfer_data, validation_data, automation_candidates, risk_register_data)

# ======================================================================================
# SECTION 4: PLOTTING & ANALYSIS FUNCTIONS
# ======================================================================================
def plot_levey_jennings(df):
    render_full_chart_briefing(context="Daily QC analysis of a certified reference material on an HPLC system to ensure system performance.", significance="Detects shifts or increased variability in an analytical instrument's performance, ensuring the validity of daily sample results. It distinguishes between random error (a single outlier) and systematic error (a developing bias).", regulatory="Directly supports **21 CFR 211.160** (General requirements for laboratory controls) and **ISO 17025** by providing documented evidence of the ongoing validity of test methods. Westgard rules are an industry best practice for clinical and QC labs.")
    baseline_points = 20
    baseline_data = df['Value'].iloc[:baseline_points]
    mean, sd = baseline_data.mean(), baseline_data.std(ddof=1)
    fig = go.Figure()
    fig.add_hrect(y0=mean - 3*sd, y1=mean + 3*sd, line_width=0, fillcolor='rgba(255, 0, 0, 0.1)', layer="below", name='±3s UCL/LCL')
    fig.add_hrect(y0=mean - 2*sd, y1=mean + 2*sd, line_width=0, fillcolor='rgba(255, 193, 7, 0.2)', layer="below", name='±2s Warning')
    fig.add_hrect(y0=mean - 1.5*sd, y1=mean + 1.5*sd, line_width=0, fillcolor='rgba(76, 175, 80, 0.1)', layer="below", name='±1.5s Guard Band')
    fig.add_trace(go.Scatter(x=df.index, y=df['Value'], mode='lines+markers', name='QC Value', line=dict(color=PRIMARY_COLOR), customdata=df['Analyst'], hovertemplate='Run: %{x}<br>Value: %{y:.2f}<br>Analyst: %{customdata}<extra></extra>'))
    fig.add_hline(y=mean, line_dash='solid', line_color=SUCCESS_GREEN, annotation_text=f"Mean ({mean:.2f})")
    violations_found = []
    violation_1_3s = df[np.abs(df['Value'] - mean) > 3 * sd]
    if not violation_1_3s.empty:
        idx = violation_1_3s.index[0]
        fig.add_annotation(x=idx, y=df['Value'].loc[idx], text="<b>1-3s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=-50, bgcolor=ERROR_RED, font=dict(color='white'))
        violations_found.append(f"1-3s failure at Run #{idx}")
    for i in range(1, len(df)):
        if (df['Value'].iloc[i-1] > mean + 2*sd and df['Value'].iloc[i] > mean + 2*sd) or \
           (df['Value'].iloc[i-1] < mean - 2*sd and df['Value'].iloc[i] < mean - 2*sd):
            idx = i
            fig.add_annotation(x=idx, y=df['Value'].loc[idx], text="<b>2-2s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=50, bgcolor=WARNING_AMBER)
            violations_found.append(f"2-2s failure at Run #{idx}")
            break
    fig.update_layout(title="<b>Levey-Jennings Chart with Guard Bands & Westgard Rules</b>", yaxis_title="Reference Material Recovery (%)", xaxis_title="Run Number")
    st.plotly_chart(fig, use_container_width=True)
    if violations_found:
        st.error(f"**Actionable Insight:** Violations detected: {'; '.join(violations_found)}. The 1-3s violation is a definitive failure. The 2-2s violation is a clear signal of systematic bias. The earlier points breaching the ±1.5s guard band served as an early warning. **Decision:** Halt testing. Initiate a formal deviation to investigate the systematic bias. Quarantine all data generated since the start of the trend.")
    else:
        st.success("**Actionable Insight:** Process appears to be in a state of statistical control. **Decision:** Continue routine testing and monitoring.")

def plot_ewma_chart(df):
    render_full_chart_briefing(context="Monitoring a critical quality attribute (CQA) where early detection of small drifts is paramount.", significance="Highly sensitive to small, persistent process drifts. This example shows an emerging drift, a corrective action, and the subsequent return to a state of control, demonstrating a full quality feedback loop.", regulatory="Supports Continued Process Verification (**FDA Process Validation Guidance**) by providing evidence of both process monitoring and the effectiveness of corrective actions. This is a key principle of **ICH Q10**.")
    lam = 0.2
    baseline_points = 15
    mean_pre, sd_pre = df['Impurity'].iloc[:baseline_points].mean(), df['Impurity'].iloc[:baseline_points].std(ddof=1)
    df['EWMA'] = df['Impurity'].ewm(span=(2/lam)-1, adjust=False).mean()
    limit_factor = 3 * sd_pre * math.sqrt(lam / (2 - lam))
    ucl = mean_pre + limit_factor
    lcl = mean_pre - limit_factor
    fig = go.Figure();
    fig.add_trace(go.Scatter(x=df.index, y=df['Impurity'], mode='markers', name='Individual Batch', marker_color=NEUTRAL_GREY, customdata=df['Batch'], hovertemplate='Batch: %{customdata}<br>Impurity: %{y:.3f}%<extra></extra>'));
    fig.add_trace(go.Scatter(x=df.index, y=df['EWMA'], mode='lines', name='EWMA', line=dict(color=PRIMARY_COLOR, width=3)));
    fig.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash='dash', line_color=ERROR_RED, annotation_text="LCL")
    fig.add_hline(y=mean_pre, line_dash='dot', line_color=SUCCESS_GREEN, annotation_text="Center Line")
    fig.add_vline(x=25, line_dash='dot', line_color=DARK_GREY, annotation_text="Process Intervention")
    violation_idx = df[df['EWMA'] > ucl].first_valid_index()
    if violation_idx is not None:
        fig.add_annotation(x=violation_idx, y=df['EWMA'][violation_idx], text="<b>Drift Signal</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title="<b>EWMA Chart with Process Intervention & Reset</b>", yaxis_title="Impurity Level (%)", xaxis_title="Batch Number")
    st.plotly_chart(fig, use_container_width=True)
    if violation_idx is not None:
        st.success(f"**Actionable Insight:** The EWMA chart correctly signaled a process drift at Batch #{violation_idx}. A planned intervention (e.g., column repacking) was performed at Batch #25. The post-intervention EWMA shows the process has returned to its original state of control. **Decision:** The corrective action was successful. Continue monitoring under the established control limits.")
    else:
        st.info("**Actionable Insight:** No significant process drift detected by the EWMA chart.")

def plot_cusum_chart(df):
    render_full_chart_briefing(context="Monitoring a high-speed, high-cost process parameter where rapid shift detection is critical.", significance="CUSUM charts are the fastest at detecting small, sustained shifts. This chart uses the tabular method with a Decision Interval (H), a formal and robust method for signal detection.", regulatory="Demonstrates an advanced and mature quality system. The use of tabular CUSUM is a highly technical SPC method that shows a deep commitment to rapid response and process control, aligning with **ICH Q9** risk-based principles.")
    baseline_points = 25
    baseline_data = df['Fill_Volume'].iloc[:baseline_points]
    target, sd = baseline_data.mean(), baseline_data.std(ddof=1)
    k = 0.5 * sd
    h = 5 * sd
    c_plus_list, c_minus_list = [], []
    c_plus_val, c_minus_val = 0.0, 0.0
    for val in df['Fill_Volume']:
        c_plus_val = max(0, val - (target + k) + c_plus_val)
        c_minus_val = max(0, (target - k) - val + c_minus_val)
        c_plus_list.append(c_plus_val)
        c_minus_list.append(c_minus_val)
    df['C+'] = c_plus_list
    df['C-'] = c_minus_list
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("CUSUM for Upward Shift (C+)", "CUSUM for Downward Shift (C-)"))
    fig.add_trace(go.Scatter(y=df['C+'], name='CUSUM High (C+)', mode='lines+markers', line=dict(color=PRIMARY_COLOR)), row=1, col=1)
    fig.add_hline(y=h, line_dash='dash', line_color=ERROR_RED, annotation_text=f"H ({h:.2f})", row=1, col=1)
    fig.add_trace(go.Scatter(y=df['C-'], name='CUSUM Low (C-)', mode='lines+markers', line=dict(color=WARNING_AMBER)), row=2, col=1)
    fig.add_hline(y=h, line_dash='dash', line_color=ERROR_RED, annotation_text=f"H ({h:.2f})", row=2, col=1)
    violation_idx = df[df['C-'] > h].first_valid_index()
    if violation_idx is not None:
        fig.add_annotation(x=violation_idx, y=df['C-'][violation_idx], text="<b>CUSUM Signal!</b>", showarrow=False, bgcolor=ERROR_RED, font=dict(color='white'), yshift=10, row=2, col=1)
    fig.update_layout(height=500, title_text="<b>Two-Sided Tabular CUSUM Chart</b>", showlegend=False)
    fig.update_yaxes(title_text="CUSUM Value", row=1, col=1)
    fig.update_yaxes(title_text="CUSUM Value", row=2, col=1)
    fig.update_xaxes(title_text="Sample Number", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    if violation_idx is not None:
        st.success(f"**Actionable Insight:** The two-sided CUSUM chart detected a downward shift at sample #{violation_idx}. The C- statistic crossed the decision interval H, providing a statistically sound signal of a process change. **Decision:** The filling line was halted. The rapid CUSUM detection minimized the number of non-conforming units produced.")
    else:
        st.info("**Actionable Insight:** No significant process shift detected by the CUSUM chart.")

def plot_zone_chart(df):
    render_full_chart_briefing(context="Monitoring the stability of a mature, well-understood, and highly capable analytical method, like a validated potency assay.", significance="Detects unnatural, non-random patterns *within* the control limits. It provides a much earlier warning of potential process drift than a standard chart that only alarms on UCL/LCL breaches, allowing for proactive investigation.", regulatory="Demonstrates a sophisticated level of process understanding and monitoring, aligning with the principles of **ICH Q14** (Analytical Procedure Development) and Continued Process Verification. The use of sensitizing rules (e.g., Nelson, Westgard) is a hallmark of a mature quality system.")
    mean, sd = df['Seal_Strength'].mean(), df['Seal_Strength'].std(ddof=1)
    fig = go.Figure()
    zones = {'Zone A (Upper)': [mean + 2*sd, mean + 3*sd], 'Zone B (Upper)': [mean + 1*sd, mean + 2*sd], 'Zone C (Upper)': [mean, mean + 1*sd], 'Zone C (Lower)': [mean - 1*sd, mean], 'Zone B (Lower)': [mean - 2*sd, mean - 1*sd], 'Zone A (Lower)': [mean - 3*sd, mean - 2*sd]}
    colors = {'A': 'rgba(255, 193, 7, 0.2)', 'B': 'rgba(76, 175, 80, 0.2)', 'C': 'rgba(76, 175, 80, 0.1)'}
    for name, y_range in zones.items():
        zone_letter = name.split(' ')[1]
        fig.add_hrect(y0=y_range[0], y1=y_range[1], line_width=0, fillcolor=colors[zone_letter], annotation_text=f"Zone {zone_letter}", annotation_position="top left", layer="below")
    fig.add_trace(go.Scatter(y=df['Seal_Strength'], mode='lines+markers', name='Strength', line=dict(color=PRIMARY_COLOR), customdata=df['Operator'], hovertemplate='Strength: %{y:.2f} N<br>Operator: %{customdata}<extra></extra>'))
    fig.add_hline(y=mean, line_color='black')
    for i in range(8, len(df)):
        if all(df['Seal_Strength'][i-8:i] > mean) or all(df['Seal_Strength'][i-8:i] < mean):
            fig.add_annotation(x=i-4, y=df['Seal_Strength'][i-4], text="<b>Rule Violation!</b><br>8 consecutive points<br>on one side of mean.", showarrow=False, bgcolor=WARNING_AMBER, borderpad=4)
            break
    fig.update_layout(title="<b>Zone Chart for Seal Strength with Sensitizing Rules</b>", yaxis_title="Seal Strength (N)", xaxis_title="Sample Number")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Actionable Insight:** Although no single point is out of control, the Zone Chart detected a run of 8 consecutive points below the center line. This non-random pattern indicates a systematic process shift. **Decision:** This early warning triggers an investigation into potential causes like equipment wear or material changes during the next planned maintenance cycle.")

def plot_i_mr_chart(df):
    render_full_chart_briefing(context="Monitoring individual measurements where assessing both process mean and variability is critical.", significance="This enhanced I-MR chart automatically applies and visualizes Nelson Rules, detecting non-random patterns (like trends or shifts) within control limits that would be missed by simple UCL/LCL breaches, providing an earlier warning of process instability.", regulatory="Goes beyond basic SPC to demonstrate a mature process monitoring program. Using sensitizing rules like Nelson Rules is a best practice for Continued Process Verification (**FDA Guidance**) and demonstrates a proactive approach to quality management (**ICH Q10**).")
    i_data = df['Purity']
    i_mean = i_data.mean()
    mr = abs(i_data.diff())
    mr_mean = mr.mean()
    i_ucl, i_lcl = i_mean + 2.66 * mr_mean, i_mean - 2.66 * mr_mean
    mr_ucl = 3.267 * mr_mean
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart with Nelson Rules</b>", "<b>Moving Range (MR) Chart</b>"))
    fig.add_trace(go.Scatter(x=df['Date'], y=i_data, name='Purity', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1);
    fig.add_hline(y=i_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="Mean");
    fig.add_hline(y=i_ucl, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL");
    fig.add_hline(y=i_lcl, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL");
    sigma_hat = mr_mean / 1.128
    if sigma_hat > 0:
        s = (i_data - i_mean) / sigma_hat
        for j in range(9, len(s)):
            if all(s[j-9:j] < 0) or all(s[j-9:j] > 0):
                fig.add_annotation(x=df['Date'].iloc[j-4], y=i_data.iloc[j-4], text="<b>Nelson Rule 2</b><br>9 points one side", showarrow=True, bgcolor=WARNING_AMBER, row=1, col=1); break
    fig.add_trace(go.Scatter(x=df['Date'], y=mr, name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1);
    fig.add_hline(y=mr_mean, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="Mean");
    fig.add_hline(y=mr_ucl, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL");
    fig.update_layout(height=600, showlegend=False, title_text="<b>I-MR Chart for Reference Standard Purity</b>")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Actionable Insight:** Even before any points breached the control limits, the I-chart detected a violation of Nelson Rule #2 (9 consecutive points below the mean), indicating a non-random downward shift. This early warning confirms the reference standard is degrading. **Decision:** Quarantine the current standard and qualify a new one immediately. This proactive measure prevents the generation of invalid data.")

def plot_cpk_analysis(df):
    render_full_chart_briefing(context="Assessing if a validated manufacturing process can reliably meet not just its official specifications, but also its tighter internal 'guard band' limits.", significance="Introduces **Guard-Banded Cpk (Cpk-GB)**, a critical internal metric that measures process capability against tighter, action-oriented limits. This provides an early warning if a process is drifting towards an edge of the specification, even if it's still officially 'in-spec'.", regulatory="Demonstrates a mature, risk-based approach to process control (**ICH Q9**). Maintaining a high Cpk-GB ensures the process stays well within the 'safe' operating space defined in the Design Space (**ICH Q8**), reducing the risk of OOS results.")
    data = df['Titer']
    LSL, USL, target = 48.0, 52.0, 50.0
    GBL, GBU = 49.0, 51.0
    mu, std = data.mean(), data.std(ddof=1)
    if std > 1e-9:
        cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std))
        cp = (USL - LSL) / (6 * std)
        cpk_gb = min((GBU - mu) / (3 * std), (mu - GBL) / (3 * std))
    else:
        cpk, cp, cpk_gb = float('inf'), float('inf'), float('inf')
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
    data = df.values
    mean_vec = data.mean(axis=0)
    inv_cov_mat = np.linalg.inv(np.cov(data, rowvar=False))
    t_squared_values = np.array([ (x - mean_vec).T @ inv_cov_mat @ (x - mean_vec) for x in data ])
    p = data.shape[1]
    m = data.shape[0]
    alpha = 0.99 
    f_critical_value = stats.f.ppf(alpha, p, m - p)
    ucl_factor = (p * (m - 1) * (m + 1)) / (m * (m - p))
    ucl = ucl_factor * f_critical_value
    anomaly_idx = np.argmax(t_squared_values)
    col1, col2 = st.columns(2)
    with col1:
        fig_t2 = go.Figure();
        fig_t2.add_trace(go.Scatter(y=t_squared_values, mode='lines+markers', name='T² Value', line=dict(color=PRIMARY_COLOR)));
        fig_t2.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text=f"UCL ({alpha*100}%)");
        if t_squared_values[anomaly_idx] > ucl:
            fig_t2.add_annotation(x=anomaly_idx, y=t_squared_values[anomaly_idx], text="<b>Multivariate Anomaly</b>", showarrow=True, bgcolor=ERROR_RED, font=dict(color='white'))
        fig_t2.update_layout(title="<b>Hotelling's T² Chart</b>", yaxis_title="T² Statistic", xaxis_title="Batch Number")
        st.plotly_chart(fig_t2, use_container_width=True)
    with col2:
        contributions = (data[anomaly_idx] - mean_vec) * (inv_cov_mat @ (data[anomaly_idx] - mean_vec))
        contrib_df = pd.DataFrame({'Variable': df.columns, 'Contribution': contributions})
        fig_contrib = px.bar(contrib_df, x='Variable', y='Contribution', title=f"<b>Contribution to Anomaly at Batch {anomaly_idx}</b>", color='Variable', color_discrete_map={'Purity_Pct': PRIMARY_COLOR, 'Titer (e12 vg/mL)': WARNING_AMBER})
        st.plotly_chart(fig_contrib, use_container_width=True)
    st.error(f"**Actionable Insight:** The T² chart identified a multivariate anomaly at Batch #{anomaly_idx}. While two separate charts might have missed it, the T² chart detected the unusual combination of parameters. The contribution plot clearly shows that the **{contrib_df.loc[contrib_df['Contribution'].idxmax(), 'Variable']}** was the primary driver of the deviation. **Decision:** Quarantine Batch #{anomaly_idx}. The investigation should immediately focus on the root causes related to the anomalous variable.")

def plot_p_chart(df):
    render_full_chart_briefing(context="Monitoring the proportion of failures when the sample size varies per period (e.g., monthly SST failures).", significance="This p-chart is enhanced with **Wilson Score Intervals** for each point, providing a more accurate representation of uncertainty than standard Shewhart chart limits, especially when failure rates are low. This prevents overreaction to random noise.", regulatory="Demonstrates a statistically superior method for handling proportional data, aligning with expectations for robust data analysis in quality systems (**21 CFR 211.165(d)**) and showing a deeper understanding of statistical theory.")
    df['proportion'] = df['SSTs_Failed'] / df['SSTs_Run']
    p_bar = df['SSTs_Failed'].sum() / df['SSTs_Run'].sum()
    df['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run'])
    df['LCL'] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df['SSTs_Run'])).clip(lower=0)
    intervals = [wilson_score_interval(p, n) for p, n in zip(df['proportion'], df['SSTs_Run'])]
    df['ci_low'] = [i[0] for i in intervals]; df['ci_high'] = [i[1] for i in intervals]
    fig = go.Figure();
    fig.add_trace(go.Scatter(x=df['Month'], y=df['proportion'], name='Proportion Failed', mode='lines+markers', line_color=PRIMARY_COLOR, error_y=dict(type='data', symmetric=False, array=df['ci_high']-df['proportion'], arrayminus=df['proportion']-df['ci_low'], color=GUARD_BAND_COLOR, thickness=1.5)));
    fig.add_trace(go.Scatter(x=df['Month'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash')));
    fig.add_trace(go.Scatter(x=df['Month'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'), showlegend=False))
    fig.add_hline(y=p_bar, name='Average Fail Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    fig.update_layout(title='<b>p-Chart for SST Failure Rate with Wilson Score Intervals</b>', yaxis_title='Proportion of SSTs Failed', xaxis_title='Month', yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)
    st.error("**Actionable Insight:** The p-chart reveals a statistically significant spike in the SST failure rate in October, with the point clearly breaching the dynamic UCL. The Wilson Score Interval for that point is also entirely above the average failure rate, confirming the significance of the event. **Decision:** Launch an investigation focused on events in October. Review all column changes, mobile phase preparations, and instrument maintenance records from that period to find the root cause.")

def plot_np_chart(df):
    render_full_chart_briefing(context="Weekly monitoring of the number of defective vials from a filling line, where each sample consists of a fixed number of units.", significance="The np-chart is a simple and effective tool for operators to track the absolute count of non-conforming units when the sample size is constant. It provides an immediate, easy-to-understand signal of a potential process shift.", regulatory="This chart is a fundamental tool for process monitoring under GMP and supports batch record review and quality oversight as required by **21 CFR 211.110** (Sampling and testing of in-process materials and drug products).")
    n = df['Batches_Sampled'].iloc[0]; p_bar = df['Defective_Vials'].sum() / (len(df) * n); ucl = n * p_bar + 3 * np.sqrt(n * p_bar * (1-p_bar)); lcl = max(0, n * p_bar - 3 * np.sqrt(n * p_bar * (1-p_bar)))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Week'], y=df['Defective_Vials'], name='Defective Vials', mode='lines+markers')); fig.add_hline(y=ucl, name='UCL', line_dash='dash', line_color=ERROR_RED, annotation_text="UCL"); fig.add_hline(y=lcl, name='LCL', line_dash='dash', line_color=ERROR_RED, annotation_text="LCL"); fig.add_hline(y=n*p_bar, name='Center Line', line_dash='dot', line_color=SUCCESS_GREEN, annotation_text="Mean")
    violations = df[df['Defective_Vials'] > ucl]
    if not violations.empty:
        violation_idx = violations.index[0]
        fig.add_annotation(x=df['Week'][violation_idx], y=df['Defective_Vials'][violation_idx], text="<b>Process Shift!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title='<b>np-Chart for Number of Defective Vials</b>', yaxis_title=f'Count of Defective Vials (n={n})', xaxis_title='Week'); st.plotly_chart(fig, use_container_width=True)
    st.error(f"**Actionable Insight:** The np-chart shows a statistically significant increase in the number of defective vials in Week #{violations.iloc[0]['Week'] if not violations.empty else 'N/A'}. This indicates a special cause of variation in the filling and capping process. **Decision:** Place all batches manufactured in that week on quality hold. Initiate a formal investigation focusing on the maintenance records and performance of the vial capping machine during that specific period.")

def plot_c_chart(df):
    render_full_chart_briefing(context="Routine environmental monitoring (EM) of a GMP cleanroom, counting colony-forming units (CFUs) on a settle plate.", significance="The c-chart is the standard tool for monitoring the number of occurrences (defects) in a constant area of opportunity. It tracks the background microbial level of the manufacturing environment, serving as a critical leading indicator of potential contamination risks.", regulatory="An environmental monitoring program is a cornerstone of sterile manufacturing, mandated by **EU GMP Annex 1** and FDA guidance. This chart provides the documented evidence of the state of control of the classified areas.")
    c_bar = df['Contaminants_per_Plate'].mean(); ucl = c_bar + 3 * np.sqrt(c_bar); lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Week'], y=df['Contaminants_per_Plate'], name='Contaminants', mode='lines+markers')); fig.add_hline(y=ucl, name='UCL', line_dash='dash', line_color=ERROR_RED, annotation_text="UCL"); fig.add_hline(y=lcl, name='LCL', line_dash='dash', line_color=ERROR_RED, annotation_text="LCL"); fig.add_hline(y=c_bar, name='Center Line', line_dash='dot', line_color=SUCCESS_GREEN, annotation_text="Mean")
    violations = df[df['Contaminants_per_Plate'] > ucl]
    if not violations.empty:
        violation_idx = violations.index[0]
        fig.add_annotation(x=df['Week'][violation_idx], y=df['Contaminants_per_Plate'][violation_idx], text="<b>Action Limit Exceeded!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title='<b>c-Chart for Environmental Monitoring</b>', yaxis_title='Colony Count per Plate', xaxis_title='Week'); st.plotly_chart(fig, use_container_width=True)
    st.error(f"**Actionable Insight:** An action limit was exceeded in Week #{violations.iloc[0]['Week'] if not violations.empty else 'N/A'}, indicating a loss of environmental control. This is a significant quality event that could impact product sterility. **Decision:** Launch a high-priority investigation. Review all activities from that week, including HVAC performance data, personnel access logs, and cleaning records. All batches manufactured under these conditions must be assessed for potential impact.")

def plot_u_chart(df):
    render_full_chart_briefing(context="Monitoring the rate of particulate defects found during visual inspection of finished vials, where the sample size per batch varies.", significance="The u-chart is essential when the sample size or 'area of opportunity' changes for each subgroup. It normalizes the defect count into a rate (defects per unit), allowing for a fair, apples-to-apples comparison of process performance across batches of different sizes.", regulatory="Demonstrates a more sophisticated level of statistical control by properly accounting for varying sample sizes. This aligns with the principles of robust data analysis expected for batch release testing under **21 CFR 211.165**.")
    df['defects_per_unit'] = df['Particulate_Defects'] / df['Vials_Inspected']; u_bar = df['Particulate_Defects'].sum() / df['Vials_Inspected'].sum(); df['UCL'] = u_bar + 3 * np.sqrt(u_bar / df['Vials_Inspected']); df['LCL'] = (u_bar - 3 * np.sqrt(u_bar / df['Vials_Inspected'])).clip(lower=0)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Batch'], y=df['defects_per_unit'], name='Defect Rate', mode='lines+markers')); fig.add_trace(go.Scatter(x=df['Batch'], y=df['UCL'], name='UCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'))); fig.add_trace(go.Scatter(x=df['Batch'], y=df['LCL'], name='LCL (Varying)', mode='lines', line=dict(color=ERROR_RED, dash='dash'), showlegend=False)); fig.add_hline(y=u_bar, name='Average Rate', line=dict(color=SUCCESS_GREEN, dash='dot'))
    df['is_violation'] = df['defects_per_unit'] > df['UCL']
    violations = df[df['is_violation']]
    if not violations.empty:
        violation_idx = violations.index[0]
        fig.add_annotation(x=df['Batch'][violation_idx], y=df['defects_per_unit'][violation_idx], text="<b>Abnormal Defect Rate!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
    fig.update_layout(title='<b>u-Chart for Particulate Defect Rate</b>', yaxis_title='Defects per Vial', xaxis_title='Batch'); st.plotly_chart(fig, use_container_width=True)
    st.error(f"**Actionable Insight:** The u-chart correctly identified a statistically significant spike in the defect *rate* for Batch #{violations.iloc[0]['Batch'] if not violations.empty else 'N/A'}. A simple count might have missed this, but normalizing by sample size revealed a true process anomaly for this specific batch. **Decision:** Quarantine this batch immediately. The investigation must focus on the specific manufacturing records for this batch, looking for deviations such as issues with the vial washing process or a line stoppage that could have introduced particulates.")

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
        shelf_life = 0
        if pooled_justified:
            slope, intercept, _, _, _ = stats.linregress(df['Time_months'], df['Potency_pct'])
            if slope < 0:
                shelf_life = (spec_limit - intercept) / slope
                fig.add_trace(go.Scatter(x=df['Time_months'], y=intercept + slope * df['Time_months'], name='Pooled Regression', line=dict(color='black', width=4, dash='dash')))
                fig.add_vline(x=shelf_life, line_dash='dot', line_color=SUCCESS_GREEN, annotation_text=f"Pooled Shelf Life: {shelf_life:.1f} mo")
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

def plot_backlog_forecast(df):
    render_full_chart_briefing(context="The AD Ops leader needs to plan resource allocation (headcount, instrument time) for the upcoming quarters.", significance="Moves planning from reactive to proactive. By forecasting the future sample backlog, a leader can provide data-driven justification for hiring new staff or purchasing new equipment *before* the lab becomes a bottleneck to the entire R&D organization.", regulatory="While not a direct compliance requirement, this demonstrates strong resource and capacity planning, a key competency for laboratory management under quality systems like **ISO 17025** and general GxP.")
    model = SimpleExpSmoothing(df['Backlog'], initialization_method="estimated").fit()
    forecast = model.forecast(26)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Week'], y=df['Backlog'], name='Historical Backlog', line=dict(color=PRIMARY_COLOR)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecasted Backlog', line=dict(color=ERROR_RED, dash='dash')))
    st.plotly_chart(fig, use_container_width=True)
    st.error(f"**Actionable Insight:** The time series forecast predicts that the sample backlog will exceed **{forecast.iloc[-1]:.0f} samples** within the next 6 months. This trend is unsustainable with the current staffing level. **Decision:** Submit a formal headcount request for one additional Research Associate, using this forecast as the primary data-driven justification.")

@st.cache_resource
def get_maint_model(_df):
    features = ['Run_Hours', 'Pressure_Spikes', 'Column_Age_Days']; target = 'Needs_Maint'; X, y = _df[features], _df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y); return model, features

def run_hplc_maintenance_model(df):
    render_full_chart_briefing(context="Managing a fleet of HPLC instruments.", significance="This tool provides a real-time risk score and explains which factors are most important for the model's prediction, enabling proactive and justifiable maintenance planning.", regulatory="A predictive, risk-based approach aligns with **ICH Q9**. Explaining the model's reasoning via feature importances supports a strong justification for maintenance decisions during audits, aligning with the principles of Computer Software Assurance (CSA).")
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
        st.subheader("Model Explainability: Feature Importance")
        st.info("This plot shows the factors the model weighs most heavily when calculating the risk score.")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Key Drivers of Maintenance Risk"
        )
        fig_importance.update_layout(height=300, margin=dict(t=50, b=0))
        st.plotly_chart(fig_importance, use_container_width=True)

    st.warning("**Actionable Insight:** The Feature Importance plot shows that the model's risk score is most influenced by **Run Hours** and **Column Age**. The current high values for these parameters are driving the high risk score. **Decision:** Schedule a full preventative maintenance, including pump seal replacement and a new column, to bring the risk score back into the green zone.")

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
    causes = { 'Man (Analyst)': ['Inadequate training', 'Improper sample preparation', 'Calculation error'], 'Machine (Instrument)': ['Leaky pump seal', 'Detector lamp aging', 'Incorrect injection volume'], 'Material (Reagents/Sample)': ['Reference standard degraded', 'Contaminated mobile phase', 'Sample degradation'], 'Method': ['SOP is unclear or incorrect', 'Method not robust to lab conditions', 'Incorrect column equilibration time'], 'Measurement': ['Integration parameters incorrect', 'Calibration curve expired', 'Wrong standard concentration used'], 'Environment': ['Lab temperature out of range', 'Vibration near balance', 'Power fluctuation'] }
    cols = st.columns(3)
    for i, cat in enumerate(causes.keys()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{cat}**")
                for cause in causes[cat]:
                    st.markdown(f"- {cause}")
    st.success("**Actionable Insight:** The investigation team uses this structured tool to brainstorm. After testing several hypotheses, the team confirmed through re-analysis with a freshly prepared standard that the **'Reference standard degraded'** (under 'Material') was the true root cause. **Decision:** A CAPA will be initiated to revise the reference standard management SOP to include more frequent stability checks.")

def render_troubleshooting_flowchart():
    st.info("This flowchart provides a systematic, compliant path for investigating an Out-of-Specification (OOS) result.")
    graph_definition = """digraph OOS_Flowchart { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#E3F2FD", color="#673ab7", fontname="sans-serif"]; edge [color="#455A64", fontname="sans-serif"]; subgraph cluster_phase1 { label = "Phase 1: Initial Investigation"; style="rounded,filled"; color="#F0F2F6"; oos [label="OOS Result Confirmed"]; check [label="Lab Investigation: Obvious Error Check\\n(e.g., calculation, dilution)"]; error_found [shape=diamond, label="Obvious Error Found?"]; invalidate [label="Invalidate Result (with justification)\\nRe-test per SOP"]; no_error [label="No Obvious Error Found"]; oos -> check -> error_found; error_found -> invalidate [label="Yes"]; error_found -> no_error [label="No"]; } subgraph cluster_phase2 { label = "Phase 2: Full-Scale Investigation"; style="rounded,filled"; color="#F0F2F6"; full_rca [label="Conduct Full RCA\\n(Fishbone, 5 Whys)"]; mfg [label="Review Manufacturing & Process Data"]; retest [label="Hypothesis-Driven Retesting\\n(e.g., new column, fresh reagents)"]; root_cause [shape=diamond, label="Root Cause Identified?"]; no_error -> full_rca; full_rca -> mfg; full_rca -> retest; } subgraph cluster_phase3 { label = "Phase 3: Conclusion & CAPA"; style="rounded,filled"; color="#F0F2F6"; capa [label="Implement CAPA\\n(Corrective & Preventive Action)"]; batch_decision [label="Make Batch Disposition Decision\\n(Release, Reject, Rework)"]; conclude [label="Close Investigation"]; root_cause -> capa [label="Yes"]; root_cause -> batch_decision [label="No (Inconclusive)"]; capa -> batch_decision; batch_decision -> conclude; } retest -> root_cause; mfg -> root_cause; }"""
    st.graphviz_chart(graph_definition)
    st.success("**Actionable Insight:** An OOS investigation must be a formal, documented process. This flowchart ensures all required steps are taken, from the initial check for simple errors to a full-scale RCA and CAPA implementation. **Decision:** All lab personnel will be retrained on this OOS procedure to ensure consistent and compliant execution.")

def render_technical_deep_dive_page():
    st.title("Technical Deep Dive: Key Analytical Methods")
    render_manager_briefing(
        title="Demonstrating Hands-On Technical Proficiency",
        content="This hub showcases detailed, content-rich visualizations for the key analytical techniques used to characterize biotechnology products. Each plot demonstrates not only the ability to generate data, but also to analyze it, interpret its meaning, and make sound scientific and business decisions based on the results.",
        reg_refs="ICH Q2(R1), ICH Q6B",
        business_impact="Ensures that all analytical data supporting process development and product characterization is robust, reliable, and defensible.",
        quality_pillar="Scientific Rigor & Data Integrity.",
        risk_mitigation="Early and accurate identification of product quality attribute shifts or assay performance issues."
    )
    def render_plot_wrapper(title, plot_function):
        with st.container(border=True):
            st.subheader(title, divider='violet')
            plot_function()
            st.markdown("---")
            
    render_plot_wrapper("HPLC: Stability Analysis & Shelf-Life Prediction", plot_hplc_purity_and_stability)
    render_plot_wrapper("Capillary Electrophoresis: Charge Variant Analysis", plot_capillary_electrophoresis_charge_variants)
    render_plot_wrapper("ELISA: Potency Assay Dose-Response Curve", plot_elisa_dose_response_curve)
    render_plot_wrapper("ddPCR: Absolute Quantification of Viral Titer", plot_ddpcr_quantification)

# ======================================================================================
# SECTION 5: PAGE RENDERING FUNCTIONS
# ======================================================================================
def render_executive_strategic_hub(team_df, tat_data, program_data, tech_roadmap_data, workflow_data, training_data, program_analytical_methods, risk_register_data):
    st.title("Executive & Strategic Hub")
    render_manager_briefing(
        title="Leading the AD Operations & Process Development Support Function",
        content="This hub provides a strategic, real-time overview of the AD Operations function, focusing on the core responsibilities of enabling **rapid, high-throughput testing**, managing program analytical deliverables, driving technological innovation, and developing a high-performance team. It is the central dashboard for managing the lab as a strategic business partner to Process Development.",
        reg_refs="ICH Q10, Company HR Policies, Departmental Budget",
        business_impact="Accelerates process development timelines, ensures analytical readiness for program milestones, and de-risks technology transfer to QC.",
        quality_pillar="Leadership, Operational Excellence & Strategic Planning.",
        risk_mitigation="Proactively manages team capacity, identifies program risks early, and ensures the function's technology remains state-of-the-art."
    )
    st.subheader("High-Throughput Operations KPIs", divider='violet')
    kpi1, kpi2, kpi3 = st.columns(3)
    current_tat = tat_data['TAT_Days'].iloc[-1]
    target_tat = 8.0
    kpi1.metric("Avg. Sample Turn-Around Time (Days)", f"{current_tat:.1f}", f"{(current_tat - target_tat):.1f} vs. Target {target_tat}d")
    fig_tat = go.Figure()
    fig_tat.add_trace(go.Scatter(x=tat_data['Month'], y=tat_data['TAT_Days'], mode='lines+markers', line=dict(color=PRIMARY_COLOR, shape='spline'), name='TAT'))
    fig_tat.add_hline(y=target_tat, line_dash='dash', line_color=SUCCESS_GREEN)
    fig_tat.update_layout(height=150, margin=dict(t=10, b=20, l=0, r=0), yaxis_title="Days", xaxis_title="")
    kpi1.plotly_chart(fig_tat, use_container_width=True)
    
    team_capacity = 95
    kpi2.metric("Team Utilization / Capacity", f"{team_capacity}%")
    fig_capacity = go.Figure(go.Bar(x=[team_capacity], y=['Utilization'], orientation='h', text=f"{team_capacity}%", textposition='inside', marker_color=WARNING_AMBER if team_capacity > 90 else SUCCESS_GREEN))
    fig_capacity.update_layout(xaxis=dict(range=[0, 100], showticklabels=True, title=""), yaxis=dict(showticklabels=False), height=50, margin=dict(t=0, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)')
    kpi2.plotly_chart(fig_capacity, use_container_width=True)
    kpi2.markdown(f"<small>At **{team_capacity}%**, the team is nearing full capacity. This data supports the need for an additional FTE in the next budget cycle to prevent burnout and project delays.</small>", unsafe_allow_html=True)
    
    methods_complete = 12
    methods_transferring = 3
    methods_in_dev = 5
    kpi3.metric("Methods Transferred to QC (YTD)", methods_complete, f"{methods_transferring} in progress")
    kpi3.markdown(f"**Pipeline:** `{methods_in_dev}` new methods in development for transfer in the next two quarters.")
    st.subheader("High-Throughput Testing Workflow Funnel (Weekly)", divider='violet')
    col1, col2 = st.columns([2,1])
    with col1:
        fig_funnel = go.Figure(go.Funnel(y = workflow_data['Stage'], x = workflow_data['Samples'], textinfo = "value+percent initial", marker = {"color": [PRIMARY_COLOR, PRIMARY_COLOR, PRIMARY_COLOR, PRIMARY_COLOR, WARNING_AMBER, SUCCESS_GREEN]}))
        fig_funnel.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_funnel, use_container_width=True)
    with col2:
        st.markdown("##### Analysis & Action Plan")
        st.warning("**Bottleneck Identified:** A significant drop-off (26%) occurs at the 'Data Review/Approval' stage. This is the primary driver of our current TAT.")
        st.markdown("""**Action Plan:**
        1.  **System:** Implement a peer-review system for routine results.
        2.  **Process:** Develop standardized data templates to accelerate review.
        3.  **Technology:** Prioritize LIMS integration to automate data flagging.
        **Expected Outcome:** Reduce review time by 50% and achieve target TAT of 8 days by next quarter.""")
    st.subheader("Program Leadership & Technology Roadmap", divider='violet')
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("##### **Program Pipeline Dashboard**")
        def style_status(s):
            if s == 'On Track': return f"background-color: {SUCCESS_GREEN}; color: white;"
            if s == 'Minor Delays': return f"background-color: {WARNING_AMBER}; color: black;"
            if s == 'At Risk': return f"background-color: {ERROR_RED}; color: white;"
            return ''
        st.dataframe(program_data.style.apply(lambda row: row.apply(style_status), subset=['Status']), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("##### **Analytical Program Deep Dive**")
        selected_program = st.selectbox("Select Program for Analytical Details:", program_analytical_methods.keys())
        if selected_program:
            st.markdown(f"**Key Analytical Methods for {selected_program}**")
            st.dataframe(program_analytical_methods[selected_program], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("##### **Strategic Technology & Automation Pipeline**")
        st.dataframe(tech_roadmap_data, use_container_width=True, hide_index=True)
        st.markdown("<small>**Actionable Insight:** The deployed Automated Liquid Handler is on track to meet its 18-month ROI target based on realized FTE savings. **Decision:** Use this successful ROI as a blueprint to justify the capital expenditure for the Next-Gen ddPCR platform in the upcoming budget cycle.</small>", unsafe_allow_html=True)
    st.subheader("Cross-Functional Risk & Dependency Register", divider='violet')
    render_full_chart_briefing(
        context="AD Ops is a critical partner to the Process Development (PD) teams. Our analytical timelines and capabilities directly impact their ability to execute experiments.",
        significance="This register provides a transparent, shared view of all key analytical dependencies, allowing for proactive risk management and joint problem-solving with our PD partners. It demonstrates our commitment to the overall success of the Process Development Team.",
        regulatory="A documented risk register is a key component of a mature Quality Risk Management (QRM) program as outlined in **ICH Q9**."
    )
    def style_risk(s):
        if s == 'High': return f"background-color: {ERROR_RED}; color: white;"
        if s == 'Medium': return f"background-color: {WARNING_AMBER}; color: black;"
        if s == 'Low': return f"background-color: {SUCCESS_GREEN}; color: white;"
        return ''
    st.dataframe(risk_register_data.style.apply(lambda row: row.apply(style_risk), subset=['Risk Level']), use_container_width=True, hide_index=True)
    st.error("**Actionable Insight:** The Downstream team's process development is blocked by the lack of a validated purity method for a new resin. This is the highest risk to the AAV-101 program timeline. **Decision:** Personally chair a daily stand-up with the lead scientist and Downstream representative until the bridge study is complete to ensure rapid resolution.")
    st.subheader("Team Management & GxP Compliance", divider='violet')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### **Team Competency & Strategic Development**")
        st.dataframe(team_df, use_container_width=True, hide_index=True)
        st.markdown("<small>**Actionable Insight:** The AAV-101 program requires deep ddPCR expertise, which is currently a single point of failure with S. Smith. **Decision:** Prioritize M. Lee's cross-training on ddPCR, as outlined in the development plan, to mitigate this risk.</small>", unsafe_allow_html=True)
    with col2:
        st.markdown("##### **Training & Compliance Status**")
        fig_training = px.bar(training_data, y='Module', x='Team Completion', orientation='h', text='Team Completion')
        fig_training.update_traces(texttemplate='%{x}%', textposition='inside', marker_color=PRIMARY_COLOR)
        fig_training.update_layout(xaxis_title="Completion Rate (%)", yaxis_title="", xaxis=dict(range=[0, 100]), height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_training, use_container_width=True)
        st.warning("**Actionable Insight:** The training completion for 'SOP-205: ddPCR Data Analysis' is only at 50%. This is a compliance risk and a bottleneck for the AAV-101 program. **Decision:** Schedule a mandatory training session for all relevant personnel by the end of next week. Update training records immediately upon completion to be inspection-ready.")

def render_statistical_toolkit_page(lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df):
    st.title("Advanced Statistical Toolkit")
    render_manager_briefing(title="Applying Statistical Rigor to Analytical Problems", content="This hub serves as a comprehensive toolkit for applying Statistical Process Control (SPC) to both analytical and manufacturing data, enabling robust monitoring and problem-solving.", reg_refs="ICH Q9, 21 CFR 211.165(d)", business_impact="Ensures decisions are based on objective statistical evidence rather than intuition, leading to more reliable processes and fewer quality events.", quality_pillar="Statistical Thinking & Data Literacy.", risk_mitigation="Detects process drifts and capability issues early, before they result in Out-of-Specification (OOS) results.")
    tab1, tab2, tab3 = st.tabs(["**📊 Monitoring Process Stability & Drift**", "**📈 Monitoring Quality & Yield (Attribute Data)**", "**🔎 Advanced Process & Method Insights**"])
    with tab1: st.subheader("Tools for Monitoring Continuous Data", divider='violet'); plot_i_mr_chart(imr_df); plot_levey_jennings(lj_df); plot_ewma_chart(ewma_df); plot_cusum_chart(cusum_df); plot_zone_chart(zone_df)
    with tab2: st.subheader("Tools for Monitoring Attribute (Count/Fail) Data", divider='violet'); plot_p_chart(p_df); plot_np_chart(np_df); plot_c_chart(c_df); plot_u_chart(u_df)
    with tab3: st.subheader("Tools for Deeper Process Understanding", divider='violet'); plot_hotelling_t2_chart(t2_df); plot_cpk_analysis(cpk_df)

def render_lifecycle_hub_page(stability_df, tost_df, screening_df, doe_df, transfer_data, validation_data):
    st.title("Method & Product Lifecycle Hub")
    render_manager_briefing(title="Guiding Methods from R&D to Commercial Launch", content="This hub demonstrates the strategic oversight of the entire analytical and product lifecycle, from initial development using QbD principles to post-approval changes and stability monitoring.", reg_refs="ICH Q1E, Q12, Q8/Q14", business_impact="Creates robust methods that are fit for purpose, accelerates development timelines, and simplifies post-approval changes by building deep process understanding.", quality_pillar="Lifecycle Management & Scientific Rigor.", risk_mitigation="Front-loads risk management into the development phase, reducing the likelihood of late-stage failures during validation or routine use.")
    st.subheader("Early-Stage: Quality by Design (QbD) for Method Development", divider='violet'); render_doe_suite(screening_df, doe_df)
    
    st.subheader("Method Validation & Robustness Summary", divider='violet')
    render_full_chart_briefing(
        context="Before a method can be transferred or used for GxP testing, it must be formally validated to demonstrate it is fit for purpose.",
        significance="This dashboard summarizes the key performance characteristics from validation studies against pre-defined acceptance criteria, providing a clear pass/fail status and ensuring regulatory compliance.",
        regulatory="Method validation is a mandatory cGMP requirement detailed in **ICH Q2(R1)**. This summary provides objective evidence of compliance for audits and regulatory filings."
    )
    with st.expander("Show Detailed Validation Data"):
        st.dataframe(validation_data, use_container_width=True, hide_index=True)
    summary = validation_data.groupby('Method')['Status'].apply(lambda x: (x == 'Pass').all()).reset_index()
    summary['Status Icon'] = summary['Status'].apply(lambda x: '✅' if x else '❌')
    summary['Overall Status'] = summary['Status'].apply(lambda x: 'Validated' if x else 'Validation Failed')
    st.dataframe(summary[['Method', 'Overall Status', 'Status Icon']], use_container_width=True, hide_index=True)

    st.subheader("Method Transfer Readiness Dashboard (AD to QC)", divider='violet')
    render_full_chart_briefing(context="As the 'Sending Unit', AD Ops is responsible for ensuring a seamless and successful transfer of validated methods to the 'Receiving Unit' (QC).", significance="This dashboard provides a real-time, task-level view of all ongoing method transfers. It allows for proactive risk identification and resource allocation to prevent delays in program timelines.", regulatory="A well-documented and controlled method transfer process is a critical component of **ICH Q10** and is essential for maintaining a state of cGMP compliance.")
    program_to_view = st.selectbox("Select Program to View Transfer Status:", transfer_data['Program'].unique())
    display_df = transfer_data[transfer_data['Program'] == program_to_view]
    def style_transfer_status(s):
        if s == '✅ Done': return f"background-color: {SUCCESS_GREEN}; color: white;"
        if s == 'In Progress': return f"background-color: {LIGHT_BLUE};"
        if s == 'Not Started': return f"background-color: {NEUTRAL_GREY};"
        if s == 'At Risk': return f"background-color: {ERROR_RED}; color: white;"
        return ''
    st.dataframe(display_df.style.apply(lambda row: row.apply(style_transfer_status), subset=['Status']), use_container_width=True, hide_index=True)
    st.error("**Actionable Insight:** The transfer protocol for the mAb-202 Charge Variant assay is 'At Risk' due to competing priorities for the lead scientist. This will delay the start of QC validation. **Decision:** Assign a junior scientist to complete the first draft based on the validation report, freeing up the lead to review and finalize. This mitigates the immediate risk to the program timeline.")
    st.subheader("Late-Stage: Commercial & Post-Approval Support", divider='violet'); plot_stability_analysis(stability_df); plot_method_equivalency_tost(tost_df)

def render_predictive_hub_page(oos_df, backlog_df, maintenance_df, automation_candidates):
    st.title("Predictive Operations & Diagnostics")
    render_manager_briefing(title="Building a Proactive, Data-Driven Operations Function", content="This hub showcases a forward-looking leadership approach that uses data science and machine learning to move from reactive problem-solving to proactive planning and risk mitigation.", reg_refs="ICH Q9, FDA's CSA Guidance", business_impact="Maximizes instrument uptime, accelerates investigations, and allows for data-driven resource planning, ultimately reducing operational costs and timelines.", quality_pillar="Predictive Analytics & Continuous Improvement.", risk_mitigation="Anticipates future bottlenecks, equipment failures, and quality issues before they can impact the organization.")
    
    tab1, tab2, tab3 = st.tabs(["**📈 Proactive Planning**", "**🔬 Predictive Diagnostics**", "**🤖 Automation Pipeline**"])
    with tab1:
        st.subheader("Proactive Resource & Maintenance Planning")
        plot_backlog_forecast(backlog_df)
        run_hplc_maintenance_model(maintenance_df)
    with tab2:
        st.subheader("Predictive Diagnostics & Troubleshooting")
        run_oos_prediction_model(oos_df)
    with tab3:
        st.subheader("Lab of the Future: Automation Candidate Matrix")
        render_full_chart_briefing(
            context="To build a high-throughput lab, we must systematically identify and prioritize automation opportunities.",
            significance="This matrix scores potential automation projects based on their potential to increase throughput and reduce human error. It provides a data-driven basis for our technology roadmap and capital expenditure requests.",
            regulatory="Reducing manual steps and error rates through automation is a key enabler of Data Integrity (**ALCOA+**) and is strongly encouraged by regulators."
        )
        st.dataframe(automation_candidates.sort_values('Automation Priority Score', ascending=False), use_container_width=True, hide_index=True)
        st.success("**Actionable Insight:** The ELISA for mAb-202 has the highest Automation Priority Score due to its low manual throughput and high error rate. **Decision:** Initiate a formal evaluation of automated ELISA platforms this quarter. This aligns with our Technology Roadmap and will be our top capital request for the next fiscal year.")

def render_qbd_quality_systems_hub_page(sankey_df):
    st.title("QbD & Quality Systems Hub")
    render_manager_briefing(title="Integrating Quality Systems into Analytical Development", content="This hub demonstrates a deep understanding of modern quality systems and how to embed them into the daily work of an analytical development lab, ensuring compliance and scientific excellence.", reg_refs="ICH Q8, Q9, Q10; 21 CFR 820.30; 21 CFR 211.192", business_impact="Creates more robust methods, reduces validation failures, and ensures investigations are thorough, compliant, and effective.", quality_pillar="Proactive Quality & Systematic Problem Solving.", risk_mitigation="Moves the function from a 'test-and-fix' mentality to a 'design-and-understand' paradigm, reducing overall compliance risk.")
    st.subheader("Proactive Quality by Design (QbD) & Design Controls", divider='violet'); render_qbd_sankey_chart(sankey_df); render_method_v_model()
    st.subheader("Reactive Problem Solving & Root Cause Analysis (RCA)", divider='violet'); run_interactive_rca_fishbone(); render_troubleshooting_flowchart()
# ======================================================================================
# SECTION 6: MAIN APP ROUTER (SIDEBAR NAVIGATION)
# ======================================================================================
st.sidebar.title("AD Ops Navigation")
PAGES = { 
    "Executive & Strategic Hub": render_executive_strategic_hub, 
    "Method & Product Lifecycle Hub": render_lifecycle_hub_page, 
    "Technical Deep Dive": render_technical_deep_dive_page,
    "Advanced Statistical Toolkit": render_statistical_toolkit_page, 
    "Predictive Operations & Diagnostics": render_predictive_hub_page, 
    "QbD & Quality Systems Hub": render_qbd_quality_systems_hub_page, 
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

(team_df, lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, 
 p_df, np_df, c_df, u_df, stability_df, tost_df, screening_df, doe_df, 
 oos_df, backlog_df, maintenance_df, sankey_df, 
 tat_data, program_data, tech_roadmap_data, workflow_data, training_data, 
 program_analytical_methods, transfer_data, validation_data, automation_candidates, risk_register_data) = generate_master_data()

page_function = PAGES[selection]
if selection == "Executive & Strategic Hub": 
    page_function(team_df, tat_data, program_data, tech_roadmap_data, workflow_data, training_data, program_analytical_methods, risk_register_data)
elif selection == "QbD & Quality Systems Hub": 
    page_function(sankey_df)
elif selection == "Method & Product Lifecycle Hub": 
    page_function(stability_df, tost_df, screening_df, doe_df, transfer_data, validation_data)
elif selection == "Predictive Operations & Diagnostics": 
    page_function(oos_df, backlog_df, maintenance_df, automation_candidates)
elif selection == "Advanced Statistical Toolkit": 
    page_function(lj_df, ewma_df, cusum_df, zone_df, imr_df, cpk_df, t2_df, p_df, np_df, c_df, u_df)
elif selection == "Technical Deep Dive": 
    page_function()

st.sidebar.markdown("---"); st.sidebar.markdown("### Role Focus"); st.sidebar.info("This portfolio is for an **Associate Director, Analytical Development Operations** role, demonstrating leadership in building high-throughput testing functions, managing the method lifecycle, and applying advanced statistical methods."); st.sidebar.markdown("---"); st.sidebar.markdown("### Key Regulatory & Quality Frameworks")
with st.sidebar.expander("View Applicable Guidelines", expanded=False): st.markdown("- **ICH Q1E, Q2, Q8, Q9, Q10, Q12, Q14**\n- **21 CFR Parts 11, 211, 820.30**\n- **EudraLex Vol. 4, Annex 1 & 15**\n- **ISO 17025, ISO 13485**")
