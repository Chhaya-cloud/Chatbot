import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("rf_model.pkl")

# App title
st.title("Car Price Prediction App")

# Collect inputs in correct order
hp = st.number_input("Horse Power (HP)", min_value=0.0, format="%.2f")
sp = st.number_input("Top Speed (SP)", min_value=0.0, format="%.2f")
vol = st.number_input("Engine Volume (VOL)", min_value=0.0, format="%.2f")
wt = st.number_input("Vehicle Weight (WT)", min_value=0.0, format="%.2f")

# When user clicks 'Predict'
if st.button("Predict Price"):
    try:
        # Create dataframe with EXACT feature names and order
        input_df = pd.DataFrame({
            "HP": [hp],
            "SP": [sp],
            "VOL": [vol],
            "WT": [wt]
        })

        # Prediction
        prediction = model.predict(input_df)[0]

        # Display result
        st.success(f"Estimated Car Price: ₹{round(prediction, 2)} lakhs")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

