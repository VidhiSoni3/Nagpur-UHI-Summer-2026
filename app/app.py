import streamlit as st
import pandas as pd
import joblib
import os
import shap

st.set_page_config(
page_title="Nagpur Urban Heat Island Analysis",
page_icon="🌡️",
layout="centered"
)

MODEL_PATH = os.path.join(
os.path.dirname(**file**),
"..",
"model",
"Nagpur_UHI_RandomForest_2026_small.pkl"
)

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

REFERENCE_PATH = os.path.join(
os.path.dirname(**file**),
"..",
"data",
"Nagpur_UHI_Daily_Mean_LST_2026.csv"
)

daily_lst = pd.read_csv(REFERENCE_PATH)
daily_lst["Date"] = pd.to_datetime(daily_lst["Date"])

st.markdown(
""" <div style="padding: 0.5rem 0 1.5rem 0;"> <h1 style="margin-bottom: 0.2rem;">
Nagpur Urban Heat Island Analysis </h1> <p style="font-size: 1.05rem; color: #666;">
Machine-learning based Land Surface Temperature analysis
for Nagpur, Maharashtra </p> </div>
""",
unsafe_allow_html=True
)

st.caption(
"Study period: March–May 2026 | "
"Model: Random Forest Regression | "
"Predictors: NDVI, NDBI, T2M"
)

st.divider()

st.subheader("Observation Date")

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

st.subheader("Environmental Inputs")

col1, col2, col3 = st.columns(3)

with col1:
ndvi = st.number_input(
"NDVI",
min_value=-1.0,
max_value=1.0,
value=0.20,
step=0.01,
help="Normalized Difference Vegetation Index"
)

with col2:
ndbi = st.number_input(
"NDBI",
min_value=-1.0,
max_value=1.0,
value=0.10,
step=0.01,
help="Normalized Difference Built-up Index"
)

with col3:
t2m = st.number_input(
"Air Temperature (°C)",
min_value=0.0,
max_value=60.0,
value=35.0,
step=0.1,
help="NASA POWER 2-metre air temperature"
)

st.caption(
"Enter environmental conditions for the selected observation date."
)

if st.button("Detect Heat Condition", type="primary"):

```
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

st.subheader("Heat-Island Assessment")

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

st.subheader("Model Explanation")

st.caption(
    "SHAP values indicate the contribution of each input variable "
    "to the individual LST prediction."
)

shap_df = pd.DataFrame({
    "Variable": input_data.columns,
    "Contribution": shap_values
})

shap_df["Effect"] = shap_df["Contribution"].apply(
    lambda x: "Higher predicted LST"
    if x > 0
    else "Lower predicted LST"
)

shap_df["Contribution"] = shap_df["Contribution"].round(3)

st.dataframe(
    shap_df,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Positive values increase the model prediction; "
    "negative values decrease it."
)

st.subheader("Interpretation")

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
```

st.divider()

st.subheader("Model Information")

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
"Nagpur UHI Summer 2026"
)


st.caption(
    "Nagpur UHI Summer 2026 | AI + Explainable Heat Analysis"
)
