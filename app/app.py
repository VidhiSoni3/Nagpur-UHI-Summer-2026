
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Nagpur UHI Prediction",
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

st.title("🌡️ Nagpur Urban Heat Island Prediction")

st.write(
    "Predict Land Surface Temperature (LST) using "
    "NDVI, NDBI and T2M."
)

st.divider()

st.subheader("Environmental Inputs")

ndvi = st.number_input(
    "NDVI",
    min_value=-1.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

ndbi = st.number_input(
    "NDBI",
    min_value=-1.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

t2m = st.number_input(
    "T2M (°C)",
    min_value=0.0,
    max_value=60.0,
    value=35.0,
    step=0.1
)

if st.button("Predict LST", type="primary"):

    input_data = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "T2M": [t2m]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Land Surface Temperature: "
        f"{prediction:.2f} °C"
    )

st.divider()

st.caption(
    "Nagpur UHI Summer 2026 | Random Forest Regression"
)
