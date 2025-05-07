import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("rf_model.pkl")

# Streamlit UI
st.title("Car Price Prediction App")

st.write("Enter the vehicle features below:")

# Input fields for each feature used in the model
hp = st.number_input("Horse Power (HP)", min_value=0.0, format="%.2f")
sp = st.number_input("Top Speed (SP)", min_value=0.0, format="%.2f")
vol = st.number_input("Engine Volume (VOL)", min_value=0.0, format="%.2f")
wt = st.number_input("Vehicle Weight (WT)", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Price"):
    try:
        # Prepare input in DataFrame
        input_data = pd.DataFrame([[hp, sp, vol, wt]], columns=["HP", "SP", "VOL", "WT"])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Show result
        st.success(f"Estimated Car Price: ₹{round(prediction, 2)} lakhs")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

