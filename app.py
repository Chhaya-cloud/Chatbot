import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('rf_model.pkl')

st.title("Car Price Prediction App 🚗")

# Feature inputs (adjust based on your actual dataset)
year = st.text_input("Enter Year of Purchase (e.g., 2015)")
present_price = st.text_input("Enter Present Price of the car (in lakhs)")
kms_driven = st.text_input("Enter Kilometers Driven")
fuel_type = st.selectbox("Select Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Select Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Select Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Convert categorical to numerical (same as your training)
fuel_type_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_type_map = {'Dealer': 0, 'Individual': 1}
transmission_map = {'Manual': 0, 'Automatic': 1}

# Prediction button
if st.button("Predict Selling Price"):
    try:
        input_data = pd.DataFrame([[
            int(year),
            float(present_price),
            int(kms_driven),
            fuel_type_map[fuel_type],
            seller_type_map[seller_type],
            transmission_map[transmission],
            int(owner)
        ]], columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Selling Price: ₹ {round(prediction, 2)} lakhs")

    except ValueError as e:
        st.error(f"Input Error: {e}")
