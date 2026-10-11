import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nagpur Urban Heat",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f3d3e 0%, #145c5f 55%, #1f7a6e 100%);
        color: white;
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        font-weight: 750;
    }

    .hero p {
        margin: 0.5rem 0 0;
        font-size: 1.02rem;
        opacity: 0.93;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #687078;
        margin-bottom: 0.9rem;
    }

    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid #e4e8eb;
        border-radius: 14px;
        background: #ffffff;
        min-height: 105px;
    }

    .info-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.9rem;
        color: #59636b;
    }

    .info-card p {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 650;
    }

    .result-card {
        padding: 1.25rem;
        border-radius: 16px;
        background: #f7fbfa;
        border: 1px solid #dcebe7;
        text-align: center;
        margin: 0.6rem 0 1rem;
    }

    .result-label {
        color: #59636b;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
    }

    .result-category {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .risk-high {
        padding: 0.95rem 1rem;
        border-radius: 12px;
        background: #fff1ed;
        border-left: 5px solid #d95c43;
    }

    .risk-moderate {
        padding: 0.95rem 1rem;
        border-radius: 12px;
        background: #fff8e8;
        border-left: 5px solid #d99a22;
    }

    .risk-normal {
        padding: 0.95rem 1rem;
        border-radius: 12px;
        background: #edf8f3;
        border-left: 5px solid #39946a;
    }

    .heat-scale {
        margin: 1rem 0 1.2rem;
        padding: 1rem 1.1rem;
        border: 1px solid #e4e8eb;
        border-radius: 16px;
        background: #ffffff;
    }

    .heat-scale-title {
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .heat-track {
        display: flex;
        gap: 5px;
        height: 16px;
        margin-bottom: 0.55rem;
    }

    .heat-segment {
        flex: 1;
        border-radius: 8px;
        background: #e8ecee;
    }

    .heat-segment.active {
        background: #145c5f;
    }

    .heat-labels {
        display: flex;
        justify-content: space-between;
        color: #687078;
        font-size: 0.78rem;
    }

    .small-note {
        color: #6c757d;
        font-size: 0.82rem;
    }

    .footer {
        text-align: center;
        color: #7a8288;
        font-size: 0.78rem;
        padding-top: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e9ec;
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD PROJECT RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_prediction_system():
    return joblib.load("model/Nagpur_UHI_RandomForest_2026_small.pkl")

@st.cache_data
def load_daily_data():
    df = pd.read_csv("data/Nagpur_UHI_Daily_Mean_LST_2026.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

try:
    prediction_system = load_prediction_system()
    daily_lst = load_daily_data()
    explainer = shap.TreeExplainer(prediction_system)
except Exception:
    st.error("The application could not load the required project resources.")
    st.caption("Please check the project files and try again.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR — SIMPLE VIEWER INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌡️ Nagpur Urban Heat")
    st.caption("Summer 2026")

    st.markdown("---")

    st.markdown("### Study date")
    selected_date = st.selectbox(
        "Choose an observation date",
        daily_lst["Date"].dt.date.tolist(),
        index=0,
    )

    selected_date = pd.Timestamp(selected_date)
    reference_row = daily_lst[daily_lst["Date"] == selected_date]
    mean_lst = float(reference_row["Mean_LST"].iloc[0])
    st.caption(
        "The date selects the reference observation for comparison. "
        "The model estimate itself uses NDVI, NDBI and T2M inputs."
    )

    st.markdown("### Model input values")

    ndvi = st.slider(
        "🌳 Vegetation",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help="Higher values generally indicate more vegetation.",
    )

    ndbi = st.slider(
        "🏙️ Built-up intensity",
        min_value=-1.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        help="Higher values generally indicate stronger built-up characteristics.",
    )

    t2m = st.slider(
        "🌡️ T2M air temperature input (°C)",
        min_value=0.0,
        max_value=60.0,
        value=35.0,
        step=0.1,
        help="Air temperature used for the estimate.",
    )

    st.markdown("---")

    st.markdown(
        '<div class="small-note">This is a research-based estimate and should not replace field measurements.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌡️ Nagpur Urban Heat Explorer</h1>
    <p>
        Explore satellite-derived land surface temperature,
        compare daily observations, and understand environmental
        factors associated with urban heat in Nagpur.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How to use this dashboard", expanded=True):
    st.markdown("""
    1. Select an observation date in the sidebar.
    2. Adjust vegetation (NDVI), built-up intensity (NDBI),
       and air temperature (T2M).
    3. View the estimated land surface temperature.
    4. Open Data Analysis to explore the available daily records.
    5. Open Why this result? to understand the model explanation.

    **Important:** Land Surface Temperature (LST) measures the
    temperature of the land surface, such as roads and rooftops.
    It is not the same as weather-station air temperature or
    the temperature of the human body.
    """)
# ---------------------------------------------------------
# PROJECT SNAPSHOT
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h4>Study area</h4>
        <p>📍 Nagpur, Maharashtra</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h4>Study period</h4>
        <p>☀️ March – May 2026</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h4>What is estimated?</h4>
        <p>🌡️ Land Surface Temperature</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# TABS — VIEWER FRIENDLY
# ---------------------------------------------------------
tab_dashboard, tab_analysis, tab_explanation, tab_resilience, tab_about = st.tabs([
    "🏠 Heat Dashboard",
    "📊 Data Analysis",
    "💡 Why this result?",
    "🌱 Climate Resilience",
    "📖 About the Project",
])

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
def calculate_prediction():
    input_data = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "T2M": [t2m],
    })

    prediction = float(prediction_system.predict(input_data)[0])

    shap_values = explainer.shap_values(input_data)
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]
    else:
        shap_values = shap_values.reshape(-1)

    uhi_anomaly = prediction - mean_lst

    if uhi_anomaly <= -4:
        category = "Strongly Cooler"
        risk_class = "normal"
    elif uhi_anomaly <= -2:
        category = "Moderately Cooler"
        risk_class = "normal"
    elif uhi_anomaly < 2:
        category = "Near Average"
        risk_class = "normal"
    elif uhi_anomaly <= 4:
        category = "Moderately Warmer"
        risk_class = "moderate"
    else:
        category = "Strongly Warmer"
        risk_class = "high"

    return input_data, prediction, shap_values, uhi_anomaly, category, risk_class

# Always calculate from the currently selected date and sidebar inputs.
# This prevents displayed results from becoming stale after inputs change.
input_data, prediction, shap_values, uhi_anomaly, category, risk_class = (
    calculate_prediction()
)

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    st.markdown(
        '<div class="section-title">Relative temperature comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Model-estimated land surface temperature compared with the selected date’s study-area reference mean.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Land Surface Temperature</div>
            <div class="result-value">{prediction:.2f} °C</div>
            <div class="result-category">{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This is a model estimate based on the selected NDVI, NDBI and T2M inputs. "
        "It is not a direct satellite measurement of a specific location. "
        "The difference below is a contextual comparison with the selected date's "
        "study-area mean, not a verified urban–rural UHI intensity or a human heat-risk score."
    )

    category_index = {
        "Strongly Cooler": 0,
        "Moderately Cooler": 1,
        "Near Average": 2,
        "Moderately Warmer": 3,
        "Strongly Warmer": 4,
    }[category]

    segments = "".join(
        '<div class="heat-segment active"></div>' if i == category_index
        else '<div class="heat-segment"></div>'
        for i in range(5)
    )

    st.markdown(
        f"""
        <div class="heat-scale">
            <div class="heat-scale-title">Heat condition</div>
            <div class="heat-track">{segments}</div>
            <div class="heat-labels">
                <span>Cooler</span>
                <span>Near average</span>
                <span>Warmer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Selected-date reference mean", f"{mean_lst:.2f} °C")

    with m2:
        st.metric("Difference from date reference", f"{uhi_anomaly:+.2f} °C")

    with m3:
        st.metric("Selected date", selected_date.strftime("%d %b %Y"))

    st.write("")

    if risk_class == "high":
        st.markdown(
            f'<div class="risk-high"><b>🔥 Estimated LST above reference</b><br>'
            f'The estimated surface temperature is substantially above the study-area average for this date.</div>',
            unsafe_allow_html=True,
        )
    elif risk_class == "moderate":
        st.markdown(
            f'<div class="risk-moderate"><b>⚠️ Estimated LST moderately above reference</b><br>'
            f'The estimated surface temperature is above the study-area average for this date.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-normal"><b>✓ Estimate near or below reference</b><br>'
            f'The estimated surface temperature is not strongly above the study-area average for this date.</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Selected conditions")

        condition_df = pd.DataFrame({
            "Indicator": ["🌳 Vegetation", "🏙️ Built-up intensity", "🌡️ Air temperature"],
            "Value": [
                f"{ndvi:.2f}",
                f"{ndbi:.2f}",
                f"{t2m:.1f} °C",
            ],
        })

        st.dataframe(
            condition_df,
            hide_index=True,
            width="stretch",
        )

    with right:
        st.markdown("### What does this mean?")

        if uhi_anomaly > 4:
            text = "The estimated surface temperature is substantially warmer than the study-area average."
        elif uhi_anomaly > 2:
            text = "The estimated surface temperature is moderately warmer than the study-area average."
        elif uhi_anomaly >= -2:
            text = "The estimated surface temperature is close to the study-area average."
        elif uhi_anomaly >= -4:
            text = "The estimated surface temperature is moderately cooler than the study-area average."
        else:
            text = "The estimated surface temperature is substantially cooler than the study-area average."

        st.info(text)

    st.write("")
    st.markdown("### Nagpur study-area LST context")

    context_df = daily_lst.copy()
    context_df["Date"] = context_df["Date"].dt.strftime("%d %b")
    context_df = context_df.rename(columns={"Mean_LST": "Mean LST (°C)"})

    st.line_chart(
        context_df.set_index("Date")[["Mean LST (°C)"]],
        width="stretch",
    )

    st.caption(
        "The chart shows the study-area mean surface-temperature observations available in the CSV during the study period; it does not imply that every calendar day was observed."
    )

# ---------------------------------------------------------
# DATA ANALYSIS — uses only the loaded daily LST dataset
# ---------------------------------------------------------
with tab_analysis:
    st.markdown("### 📊 Land Surface Temperature Analysis")
    st.caption(
        "All statistics and charts below are calculated from the loaded CSV. "
        "They describe the available observation records only; they do not "
        "represent pixel-level hotspot analysis or model-accuracy metrics. Dates shown are those present in the CSV."
    )

    analysis_df = daily_lst.copy()
    analysis_df["Mean_LST"] = pd.to_numeric(analysis_df["Mean_LST"], errors="coerce")
    valid_lst = analysis_df["Mean_LST"].dropna()
    valid_dates = analysis_df["Date"].dropna()

    total_rows = len(analysis_df)
    missing_dates = int(analysis_df["Date"].isna().sum())
    missing_lst = int(analysis_df["Mean_LST"].isna().sum())
    duplicate_dates = int(analysis_df.loc[analysis_df["Date"].notna(), "Date"].duplicated().sum())

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("CSV records", f"{total_rows}")
    a2.metric("Valid LST values", f"{len(valid_lst)}")
    a3.metric("Missing/invalid LST", f"{missing_lst}")
    a4.metric("Duplicate dates", f"{duplicate_dates}")

    if len(valid_lst) == 0:
        st.error("No valid Mean_LST values are available for analysis.")
    else:
        st.markdown("#### Summary statistics")
        stats_df = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Minimum", "Maximum", "Standard deviation"],
            "LST (°C)": [
                valid_lst.mean(),
                valid_lst.median(),
                valid_lst.min(),
                valid_lst.max(),
                valid_lst.std() if len(valid_lst) > 1 else np.nan,
            ],
        })
        st.dataframe(
            stats_df.style.format({"LST (°C)": "{:.2f}"}, na_rep="N/A"),
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### LST trend across available observation dates")
        trend_df = analysis_df.dropna(subset=["Date", "Mean_LST"]).copy()
        trend_df = trend_df.sort_values("Date")
        if trend_df.empty:
            st.info("No rows contain both a valid date and a valid LST value.")
        else:
            chart_df = trend_df.set_index("Date")[["Mean_LST"]].rename(
                columns={"Mean_LST": "Mean LST (°C)"}
            )
            st.line_chart(chart_df, width="stretch")

        st.markdown("#### Distribution of daily mean LST")
        if len(valid_lst) >= 2:
            bin_count = min(10, max(2, int(np.sqrt(len(valid_lst)))))
            counts, edges = np.histogram(valid_lst, bins=bin_count)
            hist_df = pd.DataFrame({
                "Temperature range (°C)": [
                    f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(counts))
                ],
                "Number of records": counts,
            }).set_index("Temperature range (°C)")
            st.bar_chart(hist_df, width="stretch")
        else:
            st.info("At least two valid LST records are needed to show a distribution.")

    st.markdown("#### Data quality notes")
    if missing_dates or missing_lst or duplicate_dates:
        st.warning(
            f"Please review the source CSV: {missing_dates} missing/invalid date(s), "
            f"{missing_lst} missing/invalid LST value(s), and {duplicate_dates} duplicate date(s) found. "
            "Duplicate dates are reported, not automatically averaged or removed."
        )
    else:
        st.success("No missing dates, missing/invalid LST values, or duplicate dates were detected.")

    st.markdown("#### Available observation records")
    display_df = analysis_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, hide_index=True, width="stretch")
    st.download_button(
        "Download daily LST data (CSV)",
        data=analysis_df.to_csv(index=False).encode("utf-8"),
        file_name="nagpur_daily_lst_analysis.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# WHY THIS RESULT?
# ---------------------------------------------------------
with tab_explanation:
    st.markdown(
        '<div class="section-title">Why is the estimate warmer or cooler?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">These three conditions helped shape the estimated surface temperature.</div>',
        unsafe_allow_html=True,
    )

    explanation_df = pd.DataFrame({
        "Factor": ["🌳 Vegetation", "🏙️ Built-up intensity", "🌡️ Air temperature"],
        "Influence": shap_values,
    })

    explanation_df["Strength"] = explanation_df["Influence"].abs()
    explanation_df = explanation_df.sort_values("Strength", ascending=False)

    st.markdown("### What influenced this estimate?")

    for _, row in explanation_df.iterrows():
        factor = row["Factor"]
        influence = float(row["Influence"])

        if influence > 0:
            direction_label = "Pushed the estimate higher"
            direction_icon = "↗️"
        elif influence < 0:
            direction_label = "Pushed the estimate lower"
            direction_icon = "↘️"
        else:
            direction_label = "Little influence on this estimate"
            direction_icon = "→"

        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:0.75rem;">
                <h4>{factor}</h4>
                <p style="font-weight:700;font-size:1rem;">
                    {direction_icon} {direction_label}
                </p>
                <p style="font-weight:400;font-size:0.88rem;color:#687078;">
                    Relative influence for the selected conditions: {abs(influence):.2f} °C
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    strongest = explanation_df.iloc[0]
    strongest_name = strongest["Factor"]
    strongest_value = float(strongest["Influence"])

    if strongest_value > 0:
        main_message = (
            f"{strongest_name} was the strongest influence among the selected "
            "conditions and pushed the estimate upward."
        )
    elif strongest_value < 0:
        main_message = (
            f"{strongest_name} was the strongest influence among the selected "
            "conditions and pushed the estimate downward."
        )
    else:
        main_message = (
            f"{strongest_name} had the strongest relative influence, but its "
            "effect was close to neutral for this estimate."
        )

    st.markdown("### Main influence")
    st.info(main_message)

    st.markdown("### How to understand this")

    st.write(
        "A factor that pushes the estimate higher is associated with a warmer "
        "result for the selected conditions. A factor that pushes it lower is "
        "associated with a cooler result. These influences explain the estimate "
        "but do not prove that one factor directly causes the temperature change."
    )

    st.caption(
        "SHAP values describe how each input contributes to this model prediction "
        "relative to the model baseline. They explain model behaviour; they do not "
        "prove that a feature caused a temperature change."
    )

# ---------------------------------------------------------
# CLIMATE RESILIENCE
# ---------------------------------------------------------
with tab_resilience:
    st.markdown(
        '<div class="section-title">Climate resilience ideas</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">General planning ideas associated with the relative temperature comparison.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "These suggestions are general climate-resilience ideas. The comparison is based "
        "on a model estimate versus the selected date's study-area mean; it does not identify "
        "a verified hotspot location or replace a detailed spatial assessment."
    )

    if uhi_anomaly > 4:
        st.markdown("### 🔥 Higher-priority heat condition")
        recommendations = [
            ("🌳", "Increase urban vegetation", "Support tree cover, green spaces and connected green corridors."),
            ("🏙️", "Reduce heat-absorbing surfaces", "Consider cooler or reflective materials for suitable roofs and paved areas."),
            ("💧", "Protect cooling features", "Maintain green and blue spaces that can support local cooling."),
            ("🚶", "Increase shaded public areas", "Improve shade along pedestrian routes and frequently used outdoor spaces."),
        ]
    elif uhi_anomaly > 2:
        st.markdown("### ⚠️ Moderate heat condition")
        recommendations = [
            ("🌿", "Strengthen vegetation", "Maintain and expand vegetation where green cover is limited."),
            ("🏙️", "Monitor built-up growth", "Track changes in built-up areas and their relationship with surface temperature."),
            ("💧", "Preserve cooling features", "Protect existing green and blue spaces."),
            ("🗺️", "Monitor warmer zones", "Use repeated observations to identify areas that remain warmer over time."),
        ]
    else:
        st.markdown("### 🌱 Maintain resilience")
        recommendations = [
            ("🌳", "Protect vegetation", "Maintain existing green cover and avoid unnecessary vegetation loss."),
            ("🗺️", "Continue monitoring", "Repeated observations can help identify changing heat patterns."),
            ("🏙️", "Plan heat-sensitive development", "Consider thermal conditions when evaluating future urban growth."),
            ("💧", "Preserve cooling assets", "Protect green and blue spaces that support local resilience."),
        ]

    for icon, title, description in recommendations:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:0.7rem;">
                <h4>{icon} {title}</h4>
                <p style="font-weight:400;font-size:0.95rem;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.caption(
        "These are general research-oriented ideas, not site-specific engineering or planning instructions."
    )

# ---------------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------------
with tab_about:
    st.markdown(
        '<div class="section-title">About the project</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">A simple overview of the Nagpur Summer 2026 study.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### What is this application?

    This application estimates **Land Surface Temperature (LST)** for Nagpur
    using environmental information related to vegetation, built-up intensity
    and air temperature.

    The estimate is compared with the study-area mean recorded for the selected
    observation date. This is a contextual comparison only; it is not a location-specific
    urban–rural UHI intensity measurement or a human heat-risk score.

    ### What the indicators represent

    - **🌳 Vegetation:** indicates the presence and condition of vegetation.
    - **🏙️ Built-up intensity:** indicates the degree of built-up characteristics.
    - **🌡️ Air temperature:** represents the near-surface atmospheric temperature.

    ### Study

    **Location:** Nagpur, Maharashtra  
    **Period:** March – May 2026  
    **Focus:** Urban heat, surface temperature and climate resilience

    ### Important note

    Land Surface Temperature is different from the air temperature people
    experience. The results are research-based estimates and should be
    interpreted together with local observations and other urban information.
    The selected date is used to choose the reference mean; the prediction uses
    the NDVI, NDBI and T2M inputs supplied in the sidebar.
    """)

    st.markdown("### Relative comparison categories")

    threshold_df = pd.DataFrame({
        "Difference from date reference mean": [
            "≤ −4 °C",
            "−4 to −2 °C",
            "−2 to < +2 °C",
            "+2 to +4 °C",
            "> +4 °C",
        ],
        "Relative temperature comparison": [
            "Strongly Cooler",
            "Moderately Cooler",
            "Near Average",
            "Moderately Warmer",
            "Strongly Warmer",
        ],
    })

    st.dataframe(
        threshold_df,
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Nagpur Urban Heat · Summer 2026<br>
        AI-supported research dashboard for urban heat understanding and climate resilience
    </div>
    """,
    unsafe_allow_html=True,
)
