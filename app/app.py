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
