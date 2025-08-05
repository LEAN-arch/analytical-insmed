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
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    # SME FIX: Return the feature names along with the model for robust plotting
    return model, features

def run_hplc_maintenance_model(df):
    render_full_chart_briefing(context="Managing a fleet of heavily used HPLC instruments and deciding which ones to prioritize for preventative maintenance.", significance="Shifts maintenance from a fixed, time-based schedule to a proactive, condition-based model. It reduces unnecessary maintenance on healthy instruments and prevents unexpected failures on high-risk instruments, maximizing uptime.", regulatory="Ensures instruments remain in a qualified state as required by **21 CFR 211.160(b)**. A predictive model provides a sophisticated, risk-based approach (**ICH Q9**) to maintaining the validated state of equipment.")
    # SME FIX: Unpack both the model and the feature names
    model, feature_names = get_maint_model(df)
    col1, col2, col3 = st.columns(3); hours = col1.slider("Total Run Hours", 50, 1000, 750); spikes = col2.slider("Pressure Spikes >100psi", 0, 20, 18); age = col3.slider("Column Age (Days)", 10, 300, 280)
    input_df = pd.DataFrame([[hours, spikes, age]], columns=feature_names); pred_prob = model.predict_proba(input_df)[0][1]

    colA, colB = st.columns([1,2])
    with colA:
        fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = pred_prob * 100, title = {'text': "Maintenance Risk Score"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': ERROR_RED if pred_prob > 0.7 else WARNING_AMBER if pred_prob > 0.4 else SUCCESS_GREEN}}))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    with colB:
        # SME FIX: Replaced failing SHAP plot with a robust, built-in feature importance plot.
        st.subheader("Model Feature Importance")
        st.info("This chart shows which factors the model considers most important, on average, when predicting maintenance risk. Larger bars indicate greater influence.")
        
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)

        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Key Drivers of Maintenance Risk'
        )
        fig_importance.update_layout(yaxis_title=None, xaxis_title="Model Importance Score")
        st.plotly_chart(fig_importance, use_container_width=True)

    st.warning("**Actionable Insight:** The model predicts a very high probability that HPLC-01 requires preventative maintenance. The Feature Importance chart shows that **Run Hours** and **Pressure Spikes** are the most critical factors the model uses to assess risk. Given this instrument's high values for these parameters, it is flagged as high-risk. **Decision:** Schedule HPLC-01 for maintenance this week, prioritizing it over other instruments with lower risk scores to prevent an unexpected failure during a critical run.")

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
