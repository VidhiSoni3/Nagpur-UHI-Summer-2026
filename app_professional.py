import streamlit as st
import pandas as pd
import joblib
import shap

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nagpur Urban Heat Island Analysis",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.1rem;
        color: #172033;
    }

    .hero p {
        margin: 0.25rem 0;
        color: #596273;
        font-size: 1rem;
    }

    .section-title {
        color: #172033;
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .small-note {
        color: #687386;
        font-size: 0.88rem;
    }

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        height: 100%;
    }

    .result-label {
        color: #687386;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        color: #172033;
        font-size: 1.65rem;
        font-weight: 700;
    }

    .category {
        color: #172033;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .footer {
        color: #7a8494;
        font-size: 0.82rem;
        text-align: center;
        padding-top: 1.5rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 0.8rem 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Load model and reference data
# ---------------------------------------------------------
MODEL_PATH = "model/Nagpur_UHI_RandomForest_2026_small.pkl"
DATA_PATH = "data/Nagpur_UHI_Daily_Mean_LST_2026.csv"

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

daily_lst = pd.read_csv(DATA_PATH)
daily_lst["Date"] = pd.to_datetime(daily_lst["Date"])

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌡️ Nagpur Urban Heat Island Analysis</h1>
        <p><strong>Machine-learning based Land Surface Temperature analysis for Nagpur, Maharashtra.</strong></p>
        <p>Study period: March–May 2026 &nbsp;|&nbsp; Random Forest Regression &nbsp;|&nbsp; Predictors: NDVI, NDBI, T2M</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Analysis Settings")

    selected_date = st.selectbox(
        "Observation date",
        daily_lst["Date"].dt.date.tolist()
    )

    selected_date = pd.Timestamp(selected_date)

    reference_row = daily_lst[
        daily_lst["Date"] == selected_date
    ]

    mean_lst = reference_row["Mean_LST"].iloc[0]

    st.metric(
        "Study-area mean LST",
        f"{mean_lst:.2f} °C"
    )

    st.divider()

    st.subheader("Environmental Inputs")

    ndvi = st.number_input(
        "NDVI",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help="Normalized Difference Vegetation Index."
    )

    ndbi = st.number_input(
        "NDBI",
        min_value=-1.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        help="Normalized Difference Built-up Index."
    )

    t2m = st.number_input(
        "Air temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=35.0,
        step=0.1,
        help="NASA POWER 2-metre air temperature."
    )

    run_prediction = st.button(
        "Run Heat Analysis",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "The model predicts LST from NDVI, NDBI and T2M. "
        "The UHI anomaly is calculated relative to the "
        "study-area mean LST for the selected observation date."
    )

# ---------------------------------------------------------
# Main information row
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Study Overview</div>',
    unsafe_allow_html=True
)

overview_cols = st.columns(4)

with overview_cols[0]:
    st.metric("Observation dates", f"{len(daily_lst)}")

with overview_cols[1]:
    st.metric(
        "Lowest daily mean",
        f"{daily_lst['Mean_LST'].min():.2f} °C"
    )

with overview_cols[2]:
    st.metric(
        "Highest daily mean",
        f"{daily_lst['Mean_LST'].max():.2f} °C"
    )

with overview_cols[3]:
    st.metric(
        "Model R²",
        "0.7635"
    )

st.divider()

# ---------------------------------------------------------
# Analysis tabs
# ---------------------------------------------------------
analysis_tab, explanation_tab, methodology_tab = st.tabs(
    ["Heat Analysis", "Model Explanation", "Methodology"]
)

# ---------------------------------------------------------
# Heat Analysis
# ---------------------------------------------------------
with analysis_tab:

    st.markdown(
        '<div class="section-title">Heat-Island Assessment</div>',
        unsafe_allow_html=True
    )

    if run_prediction:

        input_data = pd.DataFrame({
            "NDVI": [ndvi],
            "NDBI": [ndbi],
            "T2M": [t2m]
        })

        prediction = model.predict(input_data)[0]

        shap_values = explainer.shap_values(input_data)

        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        shap_values = shap_values[0]

        uhi_anomaly = prediction - mean_lst

        if uhi_anomaly <= -4:
            category = "Strongly Cooler"
        elif uhi_anomaly <= -2:
            category = "Moderately Cooler"
        elif uhi_anomaly < 2:
            category = "Near Average"
        elif uhi_anomaly <= 4:
            category = "Moderately Warmer"
        else:
            category = "Strongly Warmer"

        # Result cards
        result_cols = st.columns(3)

        with result_cols[0]:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Land Surface Temperature</div>
                    <div class="result-value">{prediction:.2f} °C</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_cols[1]:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">UHI anomaly</div>
                    <div class="result-value">{uhi_anomaly:+.2f} °C</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_cols[2]:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Relative heat category</div>
                    <div class="category">{category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Interpretation
        st.subheader("Interpretation")

        if uhi_anomaly > 4:
            st.warning(
                "The predicted LST is more than 4 °C above the "
                "study-area mean for the selected observation date."
            )
        elif uhi_anomaly > 2:
            st.warning(
                "The predicted LST is moderately above the "
                "study-area mean for the selected observation date."
            )
        elif uhi_anomaly >= -2:
            st.info(
                "The predicted LST is close to the study-area "
                "mean for the selected observation date."
            )
        elif uhi_anomaly >= -4:
            st.info(
                "The predicted LST is moderately below the "
                "study-area mean for the selected observation date."
            )
        else:
            st.info(
                "The predicted LST is substantially below the "
                "study-area mean for the selected observation date."
            )

        st.caption(
            "UHI anomaly = predicted LST − study-area mean LST "
            "for the selected observation date. This is a relative "
            "LST anomaly, not a classical urban-versus-rural UHI measurement."
        )

        # Input summary
        st.subheader("Input Conditions")

        input_display = pd.DataFrame({
            "Variable": ["NDVI", "NDBI", "Air temperature"],
            "Value": [
                f"{ndvi:.2f}",
                f"{ndbi:.2f}",
                f"{t2m:.1f} °C"
            ],
            "Role": [
                "Vegetation condition",
                "Built-up surface indicator",
                "Near-surface air temperature"
            ]
        })

        st.dataframe(
            input_display,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "Select an observation date, review the environmental inputs "
            "in the sidebar, and click **Run Heat Analysis**."
        )

# ---------------------------------------------------------
# Model Explanation
# ---------------------------------------------------------
with explanation_tab:

    st.markdown(
        '<div class="section-title">Explainable AI</div>',
        unsafe_allow_html=True
    )

    if run_prediction:

        shap_df = pd.DataFrame({
            "Variable": ["NDVI", "NDBI", "T2M"],
            "SHAP contribution": shap_values
        })

        shap_df["Absolute contribution"] = (
            shap_df["SHAP contribution"].abs()
        )

        shap_df["Direction"] = shap_df["SHAP contribution"].apply(
            lambda value: (
                "Raises predicted LST"
                if value > 0
                else "Lowers predicted LST"
            )
        )

        shap_df = shap_df.sort_values(
            "Absolute contribution",
            ascending=False
        )

        st.write(
            "SHAP explains how each input contributed to this individual "
            "prediction relative to the model's baseline prediction."
        )

        chart_data = shap_df.set_index("Variable")[
            ["SHAP contribution"]
        ]

        st.bar_chart(
            chart_data,
            use_container_width=True
        )

        display_df = shap_df[
            ["Variable", "SHAP contribution", "Direction"]
        ].copy()

        display_df["SHAP contribution"] = (
            display_df["SHAP contribution"].round(3)
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Positive SHAP values increase the model prediction; "
            "negative values decrease it. SHAP describes model behavior "
            "and should not be interpreted as causal evidence."
        )

    else:
        st.info(
            "Run the heat analysis first to view the explanation for "
            "the current prediction."
        )

# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------
with methodology_tab:

    st.markdown(
        '<div class="section-title">Methodology and Model Information</div>',
        unsafe_allow_html=True
    )

    method_cols = st.columns(2)

    with method_cols[0]:
        st.subheader("Prediction model")

        st.write(
            "Random Forest Regressor trained to predict Land Surface "
            "Temperature (LST)."
        )

        st.write(
            "**Predictors:** NDVI, NDBI and T2M"
        )

        st.write(
            "**Deployment model:** 25 trees, maximum depth 15"
        )

        st.write(
            "**Train/test split:** 80% / 20%"
        )

    with method_cols[1]:
        st.subheader("Held-out test performance")

        performance = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": ["2.0355 °C", "2.6301 °C", "0.7635"]
        })

        st.dataframe(
            performance,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Performance values are from the held-out test set."
        )

    st.divider()

    st.subheader("UHI anomaly classification")

    classification = pd.DataFrame({
        "Anomaly range": [
            "≤ −4 °C",
            "−4 to −2 °C",
            "−2 to +2 °C",
            "+2 to +4 °C",
            "> +4 °C"
        ],
        "Category": [
            "Strongly Cooler",
            "Moderately Cooler",
            "Near Average",
            "Moderately Warmer",
            "Strongly Warmer"
        ]
    })

    st.dataframe(
        classification,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These thresholds are project-defined descriptive categories. "
        "They are not universal physical UHI thresholds."
    )

    st.divider()

    st.subheader("Important limitations")

    st.markdown(
        """
        - The model predicts **land surface temperature**, not air temperature.
        - The UHI anomaly is calculated relative to the **study-area mean LST on the selected date**.
        - A classical urban-versus-rural UHI intensity requires an explicit rural/reference baseline.
        - SHAP values explain model behavior; they do not establish causation.
        - What-if predictions represent model-based scenarios and should not be treated as proof of intervention effects.
        - Model predictions depend on the range and quality of the training data.
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Nagpur Urban Heat Island Summer 2026 &nbsp;|&nbsp;
        AI + Explainable Heat Analysis
    </div>
    """,
    unsafe_allow_html=True
)
