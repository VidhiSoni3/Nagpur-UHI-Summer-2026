# Nagpur Urban Heat Island Prediction — Summer 2026

## Project Overview

This project develops a machine-learning model to predict Land Surface Temperature (LST) for Nagpur, Maharashtra, India, using satellite-derived environmental variables and NASA POWER air temperature data.

The final application is built with Streamlit and uses a Random Forest regression model.

## Study Area

- Location: Nagpur, Maharashtra, India
- Approximate center: 21.15°N, 79.09°E
- Study radius: approximately 15 km
- Study period: March 1, 2026 to May 31, 2026

## Input Variables

- NDVI — Normalized Difference Vegetation Index
- NDBI — Normalized Difference Built-up Index
- T2M — NASA POWER 2-metre air temperature

## Target Variable

- LST — Land Surface Temperature (°C)

## Final Deployment Model

Random Forest Regressor

- Number of trees: 25
- Maximum tree depth: 15
- Random state: 42
- Training samples: 72,693
- Test samples: 18,174
- Model size: approximately 7.59 MB

## Model Performance

Performance on the held-out test dataset:

| Metric | Value |
|---|---:|
| MAE | 2.0355 °C |
| RMSE | 2.6301 °C |
| R² | 0.7635 |

The model explains approximately 76.35% of the variance in LST on the held-out test data.

## Original Model Comparison

The original 100-tree Random Forest achieved:

- MAE: 2.1388 °C
- RMSE: 2.7582 °C
- R²: 0.7399

The final deployment model is substantially smaller while achieving stronger performance on the same held-out test set.

## Feature Importance

For the original 100-tree model:

| Feature | Importance |
|---|---:|
| T2M | 57.11% |
| NDBI | 28.04% |
| NDVI | 14.85% |

Feature importance indicates how the trained model used the variables for prediction. It should not be interpreted as causal influence.

## Application

The Streamlit application allows users to enter:

1. NDVI
2. NDBI
3. T2M

The application then predicts LST in degrees Celsius.

## Project Structure

Nagpur_UHI_Deployment/
├── README.md
├── requirements.txt
├── app/
│   └── app.py
└── model/
    ├── Nagpur_UHI_RandomForest_2026_small.pkl
    └── Nagpur_UHI_Model_Metadata_2026.json

## UHI Analysis

A date-normalized relative LST anomaly was calculated as:

UHI Intensity = LST − Mean LST for the observation date

This provides a relative spatial heat anomaly for each observation date. It is not a classical urban-versus-rural UHI intensity measurement.

## Limitations

The deployed model uses only NDVI, NDBI and T2M. Other factors that can influence LST, such as land cover, elevation, soil moisture, atmospheric conditions and spatial/temporal effects, are not explicitly included.

Individual predictions may therefore have larger errors even though the overall test performance is reasonable.

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Deployment

This folder contains the files required to deploy the Streamlit application.
