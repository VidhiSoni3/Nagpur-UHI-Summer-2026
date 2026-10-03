import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nagpur Urban Heat Intelligence",
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
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f3d3e 0%, #145c5f 55%, #1f7a6e 100%);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 750;
    }

    .hero p {
        margin: 0.45rem 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.6rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #687078;
        margin-bottom: 0.8rem;
    }

    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid #e4e8eb;
        border-radius: 14px;
        background: #ffffff;
        min-height: 110px;
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

    .risk-high {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #fff1ed;
        border-left: 5px solid #d95c43;
    }

    .risk-moderate {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #fff8e8;
        border-left: 5px solid #d99a22;
    }

    .risk-normal {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #edf8f3;
        border-left: 5px solid #39946a;
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
# DATA / MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/Nagpur_UHI_RandomForest_2026_small.pkl")

@st.cache_data
def load_daily_data():
    df = pd.read_csv("data/Nagpur_UHI_Daily_Mean_LST_2026.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

try:
    model = load_model()
    daily_lst = load_daily_data()
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error("The application could not load its project resources.")
    st.caption("Please verify that the model and data files are present in the configured project folders.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌡️ Heat Intelligence")
    st.caption("Nagpur • Summer 2026")

    st.markdown("---")
    st.markdown("### Observation")
    selected_date = st.selectbox(
        "Observation date",
        daily_lst["Date"].dt.date.tolist(),
        index=0,
        help="Select a date from the available study observations.",
    )

    selected_date = pd.Timestamp(selected_date)
    reference_row = daily_lst[daily_lst["Date"] == selected_date]

    mean_lst = float(reference_row["Mean_LST"].iloc[0])

    st.markdown("### Environmental conditions")
    ndvi = st.slider(
        "NDVI — vegetation",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help="Higher NDVI generally represents greater vegetation presence.",
    )

    ndbi = st.slider(
        "NDBI — built-up intensity",
        min_value=-1.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        help="Higher NDBI generally indicates stronger built-up characteristics.",
    )

    t2m = st.slider(
        "Air temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=35.0,
        step=0.1,
        help="Near-surface air temperature supplied as a model predictor.",
    )

    st.markdown("---")
    analyze = st.button("🔎 Analyze heat condition", type="primary", use_container_width=True)

    st.markdown(
        '<div class="small-note">The dashboard presents model-based decision support. '
        'It does not replace field measurements or detailed urban-planning assessment.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌡️ Nagpur Urban Heat Intelligence</h1>
    <p>AI-powered Urban Heat Island Detection, Explainable Analysis & Climate Resilience</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

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
        <h4>AI approach</h4>
        <p>🌲 Random Forest Regression</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="info-card">
        <h4>Explainability</h4>
        <p>🧠 SHAP-based analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_dashboard, tab_xai, tab_resilience, tab_methodology = st.tabs([
    "🏠 Dashboard",
    "🧠 Explainable AI",
    "🌱 Climate Resilience",
    "📚 Methodology & Limitations",
])

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def calculate_prediction():
    input_data = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "T2M": [t2m],
    })

    prediction = float(model.predict(input_data)[0])

    shap_values = explainer.shap_values(input_data)
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_values = np.asarray(shap_values)

    # Handles common SHAP output shapes.
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

# Keep a useful initial state so the dashboard is informative immediately.
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = calculate_prediction()

if analyze:
    st.session_state.prediction_result = calculate_prediction()

input_data, prediction, shap_values, uhi_anomaly, category, risk_class = st.session_state.prediction_result

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    st.markdown('<div class="section-title">Heat-risk overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Model-based assessment for the selected environmental conditions.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Study-area mean LST", f"{mean_lst:.2f} °C")

    with m2:
        st.metric("Predicted LST", f"{prediction:.2f} °C")

    with m3:
        st.metric("UHI anomaly", f"{uhi_anomaly:+.2f} °C")

    with m4:
        st.metric("Heat category", category)

    st.write("")

    if risk_class == "high":
        st.markdown(
            f'<div class="risk-high"><b>🔥 Strong warming signal</b><br>'
            f'The predicted surface temperature is substantially above the selected date’s '
            f'study-area mean ({mean_lst:.2f} °C). This condition is classified as <b>{category}</b>.</div>',
            unsafe_allow_html=True,
        )
    elif risk_class == "moderate":
        st.markdown(
            f'<div class="risk-moderate"><b>⚠️ Moderate warming signal</b><br>'
            f'The predicted surface temperature is above the selected date’s study-area mean. '
            f'This condition is classified as <b>{category}</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-normal"><b>✓ Near or below average</b><br>'
            f'The predicted surface temperature is not strongly above the selected date’s '
            f'study-area mean. Classification: <b>{category}</b>.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("### Environmental condition")
        condition_df = pd.DataFrame({
            "Indicator": ["NDVI", "NDBI", "Air temperature"],
            "Selected value": [ndvi, ndbi, t2m],
            "Meaning": [
                "Vegetation condition",
                "Built-up intensity",
                "Near-surface air temperature",
            ],
        })
        st.dataframe(condition_df, hide_index=True, use_container_width=True)

    with right:
        st.markdown("### Quick interpretation")

        if uhi_anomaly > 4:
            text = (
                "The model estimates substantially warmer surface conditions than "
                "the study-area mean for this date."
            )
        elif uhi_anomaly > 2:
            text = (
                "The model estimates moderately warmer surface conditions than "
                "the study-area mean for this date."
            )
        elif uhi_anomaly >= -2:
            text = (
                "The predicted surface temperature is close to the study-area "
                "mean for the selected date."
            )
        elif uhi_anomaly >= -4:
            text = (
                "The predicted surface temperature is moderately below the "
                "study-area mean for this date."
            )
        else:
            text = (
                "The predicted surface temperature is substantially below the "
                "study-area mean for this date."
            )

        st.info(text)

        st.markdown("**What the anomaly means**")
        st.write(
            "UHI anomaly = predicted LST − study-area mean LST for the selected observation date."
        )

    st.write("")
    st.markdown("### Selected-date context")

    context_df = daily_lst.copy()
    context_df["Date"] = context_df["Date"].dt.strftime("%d %b %Y")
    context_df = context_df.rename(columns={"Mean_LST": "Mean LST (°C)"})

    st.line_chart(
        context_df.set_index("Date")[["Mean LST (°C)"]],
        use_container_width=True,
    )

    st.caption(
        "The line chart shows the available study-area mean LST observations. "
        "The highlighted prediction is calculated for the selected environmental inputs."
    )

# ---------------------------------------------------------
# XAI
# ---------------------------------------------------------
with tab_xai:
    st.markdown('<div class="section-title">Explainable AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Understand which environmental inputs influenced this individual prediction.</div>',
        unsafe_allow_html=True,
    )

    shap_df = pd.DataFrame({
        "Variable": input_data.columns,
        "SHAP contribution": shap_values,
    })

    shap_df["Direction"] = shap_df["SHAP contribution"].apply(
        lambda x: "↑ Higher predicted LST" if x > 0 else "↓ Lower predicted LST"
    )
    shap_df["Absolute influence"] = shap_df["SHAP contribution"].abs()
    shap_df = shap_df.sort_values("Absolute influence", ascending=False)

    x1, x2 = st.columns([1.05, 0.95])

    with x1:
        st.markdown("### Feature contribution")
        chart_df = shap_df.set_index("Variable")[["SHAP contribution"]].sort_values(
            "SHAP contribution"
        )
        st.bar_chart(chart_df, use_container_width=True)

    with x2:
        st.markdown("### Explanation table")
        display_df = shap_df[[
            "Variable",
            "SHAP contribution",
            "Direction",
        ]].copy()
        display_df["SHAP contribution"] = display_df["SHAP contribution"].round(3)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.write("")
    st.markdown("### How to read this")

    strongest = shap_df.iloc[0]
    strongest_name = strongest["Variable"]
    strongest_value = float(strongest["SHAP contribution"])

    if strongest_value > 0:
        explanation = (
            f"Among the displayed inputs, **{strongest_name}** has the largest "
            f"absolute contribution and is pushing the model prediction upward."
        )
    else:
        explanation = (
            f"Among the displayed inputs, **{strongest_name}** has the largest "
            f"absolute contribution and is pushing the model prediction downward."
        )

    st.info(explanation)

    st.caption(
        "SHAP contribution describes the direction and magnitude of an input's "
        "contribution to this individual model prediction. It should not be interpreted "
        "as proof of causal relationships."
    )

# ---------------------------------------------------------
# CLIMATE RESILIENCE
# ---------------------------------------------------------
with tab_resilience:
    st.markdown('<div class="section-title">Climate resilience insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Planning-oriented responses associated with the observed heat condition.</div>',
        unsafe_allow_html=True,
    )

    if uhi_anomaly > 4:
        st.markdown("### 🔥 Higher-priority heat condition")
        recommendations = [
            ("🌳", "Increase urban vegetation", "Prioritize tree cover, green spaces and connected green corridors in heat-exposed areas."),
            ("🏙️", "Reduce heat-absorbing surfaces", "Consider cool or reflective materials where appropriate for roofs and paved surfaces."),
            ("💧", "Protect blue-green infrastructure", "Maintain vegetation and water-related cooling features that can support local thermal comfort."),
            ("🚶", "Improve shaded public areas", "Increase shade along pedestrian routes and other high-use outdoor spaces."),
        ]
    elif uhi_anomaly > 2:
        st.markdown("### ⚠️ Moderate heat condition")
        recommendations = [
            ("🌿", "Strengthen vegetation", "Maintain and expand vegetation in areas with limited green cover."),
            ("🏙️", "Monitor built-up growth", "Track changes in built-up intensity and their relationship with surface temperature."),
            ("💧", "Preserve cooling features", "Protect existing green and blue infrastructure."),
            ("🗺️", "Target vulnerable zones", "Use repeated observations to identify persistent warmer locations."),
        ]
    else:
        st.markdown("### 🌱 Maintain resilience")
        recommendations = [
            ("🌳", "Protect vegetation", "Maintain existing green cover and avoid unnecessary loss of urban vegetation."),
            ("🗺️", "Continue monitoring", "Repeated observations can help identify emerging heat patterns."),
            ("🏙️", "Plan heat-sensitive development", "Consider thermal conditions when evaluating future urban growth."),
            ("💧", "Preserve cooling assets", "Protect green and blue infrastructure that supports local resilience."),
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
        "These are research-oriented resilience suggestions based on the dashboard's "
        "environmental indicators. They are not site-specific engineering or planning prescriptions."
    )

# ---------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------
with tab_methodology:
    st.markdown('<div class="section-title">Project methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A simplified view of the research workflow for presentation and viva.</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Environmental data", "NDVI • NDBI • T2M • LST"),
        ("02", "Feature preparation", "Prepare model input variables"),
        ("03", "AI prediction", "Random Forest regression"),
        ("04", "UHI assessment", "Compare predicted LST with study-area mean"),
        ("05", "Explainable AI", "SHAP contribution analysis"),
        ("06", "Climate resilience", "Translate heat conditions into planning-oriented responses"),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:0.7rem;">
                <h4>{number} · {title}</h4>
                <p style="font-weight:400;font-size:0.95rem;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Model performance")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("MAE", "2.0355 °C")
    with p2:
        st.metric("RMSE", "2.6301 °C")
    with p3:
        st.metric("R²", "0.7635")

    st.caption(
        "Held-out test performance reported for the deployed Random Forest regression model."
    )

    st.markdown("### UHI classification used in this dashboard")

    threshold_df = pd.DataFrame({
        "UHI anomaly": [
            "≤ −4 °C",
            "−4 to −2 °C",
            "−2 to < +2 °C",
            "+2 to +4 °C",
            "> +4 °C",
        ],
        "Classification": [
            "Strongly Cooler",
            "Moderately Cooler",
            "Near Average",
            "Moderately Warmer",
            "Strongly Warmer",
        ],
    })

    st.dataframe(threshold_df, hide_index=True, use_container_width=True)

    st.markdown("### Scientific cautions")

    st.warning(
        "The dashboard provides model-based decision support. Satellite-derived surface "
        "temperature is different from human-perceived air temperature, and a model prediction "
        "does not by itself establish a causal relationship between an environmental variable "
        "and urban heating."
    )

    limitations = [
        "Predictions depend on the quality and representativeness of the training data.",
        "The available study period limits how broadly seasonal conclusions can be generalized.",
        "UHI interpretation depends on the spatial and temporal resolution of the underlying observations.",
        "Model performance metrics describe the reported held-out evaluation and should not be treated as universal accuracy.",
        "Detailed field measurements and urban-planning analysis are required for site-specific decisions.",
    ]

    for item in limitations:
        st.markdown(f"• {item}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Nagpur Urban Heat Intelligence · AI + Explainable AI for Urban Heat Island Detection & Climate Resilience<br>
        Academic research dashboard · Summer 2026
    </div>
    """,
    unsafe_allow_html=True,
)
