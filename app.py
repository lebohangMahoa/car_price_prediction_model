import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate a used car's asking price using a "
    "Random Forest machine learning model."
)

# --------------------------------------------------
# 2. Load the saved model bundle
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "car_price_model.joblib"

try:
    bundle = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file not found. Place car_price_model.joblib "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]

# --------------------------------------------------
# 3. Reference year
# IMPORTANT:
# Change this to the exact year used to calculate Age
# in your training notebook.
# --------------------------------------------------
REFERENCE_YEAR = 2024  # CHANGE if your notebook used another year

# --------------------------------------------------
# 4. Check the model's saved input features
# --------------------------------------------------
required_categories = [
    "brands",
    "models",
    "transmissions",
    "owners",
    "fuel_types"
]

if not all(key in bundle for key in required_categories):
    st.error(
        "The saved model bundle is missing category lists. "
        "Please recreate the bundle in your training notebook."
    )
    st.stop()

# --------------------------------------------------
# 5. Car input form
# --------------------------------------------------
with st.form("car_form"):
    st.subheader("Enter the car details")

    brand = st.selectbox(
        "Brand",
        bundle["brands"]
    )

    car_model = st.selectbox(
        "Model",
        bundle["models"]
    )

    year = st.number_input(
        "Manufacturing year",
        min_value=1980,
        max_value=REFERENCE_YEAR,
        value=min(2020, REFERENCE_YEAR),
        step=1
    )

    km_driven = st.number_input(
        "Kilometres driven",
        min_value=0,
        max_value=2_000_000,
        value=60_000,
        step=1_000
    )

    transmission = st.selectbox(
        "Transmission",
        bundle["transmissions"]
    )

    owner = st.selectbox(
        "Owner",
        bundle["owners"]
    )

    fuel_type = st.selectbox(
        "Fuel type",
        bundle["fuel_types"]
    )

    submitted = st.form_submit_button(
        "Predict asking price",
        type="primary"
    )

# --------------------------------------------------
# 6. Prepare input and predict
# --------------------------------------------------
if submitted:
    age = REFERENCE_YEAR - int(year)

    km_per_year = km_driven / (age if age > 0 else 1)

    input_data = pd.DataFrame([{
        "Brand": brand,
        "model": car_model,
        "Year": int(year),
        "kmDriven": float(km_driven),
        "Transmission": transmission,
        "Owner": owner,
        "FuelType": fuel_type,
        "KmPerYear": float(km_per_year)
    }])

    categorical_features = [
        "Brand",
        "model",
        "Transmission",
        "Owner",
        "FuelType"
    ]

    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_encoded[numeric_features] = scaler.transform(
        input_encoded[numeric_features].astype(float)
    )

    log_prediction = model.predict(input_encoded)[0]
    predicted_price = np.exp(log_prediction)

    if np.isfinite(predicted_price) and predicted_price > 0:
        st.success(
            f"Estimated asking price: ₹{predicted_price:,.0f}"
        )

        st.caption(
            "This is a machine-learning estimate, not a guaranteed "
            "market price. Actual prices depend on the car's condition, "
            "location and other factors."
        )
    else:
        st.error(
            "The model returned an invalid prediction. "
            "Please check the input data and saved model."
        )