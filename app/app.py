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

# --------------------------------------------------
# About
# --------------------------------------------------

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

