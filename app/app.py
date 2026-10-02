
import streamlit as st
import pandas as pd
import joblib
import os
import shap

st.set_page_config(
    page_title="Nagpur AI Heat Island",
    page_icon="🌡️",
    layout="centered"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "Nagpur_UHI_RandomForest_2026_small.pkl"
)

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "Nagpur_UHI_Daily_Mean_LST_2026.csv"
)

daily_lst = pd.read_csv(REFERENCE_PATH)
daily_lst["Date"] = pd.to_datetime(daily_lst["Date"])

st.title("🌡️ Nagpur AI Heat Island Detection")

st.write(
    "AI-based prediction, heat-island anomaly detection "
    "and climate-resilience analysis for Nagpur."
)

st.info(
    "The AI model predicts Land Surface Temperature (LST). "
    "UHI anomaly is calculated relative to the mean LST "
    "observed across the study area on the selected date."
)

st.divider()

st.subheader("📅 Observation Date")

selected_date = st.selectbox(
    "Select a Landsat observation date",
    daily_lst["Date"].dt.date.tolist()
)

selected_date = pd.Timestamp(selected_date)

reference_row = daily_lst[
    daily_lst["Date"] == selected_date
]

mean_lst = reference_row["Mean_LST"].iloc[0]

st.metric(
    "Study-area Mean LST",
    f"{mean_lst:.2f} °C"
)

st.subheader("🌍 Environmental Inputs")

ndvi = st.number_input(
    "NDVI",
    min_value=-1.0,
    max_value=1.0,
    value=0.20,
    step=0.01,
    help="Normalized Difference Vegetation Index"
)

ndbi = st.number_input(
    "NDBI",
    min_value=-1.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
    help="Normalized Difference Built-up Index"
)

t2m = st.number_input(
    "T2M (°C)",
    min_value=0.0,
    max_value=60.0,
    value=35.0,
    step=0.1,
    help="NASA POWER 2-metre air temperature"
)

if st.button("🔍 Detect Heat Condition", type="primary"):

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

    st.success(
        f"Predicted LST: {prediction:.2f} °C"
    )

    st.subheader("🔥 Heat-Island Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted LST",
            f"{prediction:.2f} °C"
        )

    with col2:
        st.metric(
            "UHI Anomaly",
            f"{uhi_anomaly:+.2f} °C"
        )

    st.metric(
        "Heat Category",
        category
    )

    st.subheader("🧠 Explainable AI (SHAP)")

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "SHAP Contribution": shap_values
    })

    shap_df["Impact"] = shap_df["SHAP Contribution"].apply(
        lambda x: "Increases predicted LST"
        if x > 0
        else "Decreases predicted LST"
    )

    st.dataframe(
        shap_df,
        use_container_width=True
    )

    st.bar_chart(
        shap_df.set_index("Feature")["SHAP Contribution"]
    )

    st.caption(
        "Positive SHAP values push the model prediction higher; "
        "negative SHAP values push it lower."
    )

    st.subheader("📝 Interpretation")

    if uhi_anomaly > 4:
        st.warning(
            "The predicted LST is more than 4 °C above "
            "the study-area mean for this observation date."
        )

    elif uhi_anomaly > 2:
        st.warning(
            "The predicted LST is moderately above the "
            "study-area mean for this observation date."
        )

    elif uhi_anomaly >= -2:
        st.info(
            "The predicted LST is close to the study-area "
            "mean for this observation date."
        )

    elif uhi_anomaly >= -4:
        st.info(
            "The predicted LST is moderately below the "
            "study-area mean for this observation date."
        )

    else:
        st.info(
            "The predicted LST is substantially below the "
            "study-area mean for this observation date."
        )

    st.caption(
        "UHI anomaly = predicted LST − study-area mean LST "
        "for the selected observation date."
    )

st.divider()

st.subheader("🤖 About the AI Model")

st.write(
    "Random Forest regression model using NDVI, NDBI "
    "and T2M as predictors."
)

st.write(
    "Deployment model: 25 trees, maximum depth 15."
)

st.write(
    "Held-out test performance: "
    "MAE = 2.0355 °C | "
    "RMSE = 2.6301 °C | "
    "R² = 0.7635"
)

st.caption(
    "Nagpur UHI Summer 2026 | AI + Explainable Heat Analysis"
)
