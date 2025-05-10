import streamlit as st
import pandas as pd
import pickle

# Load the trained model and the expected feature names
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("Car Price Prediction App")

# Create input fields dynamically based on saved feature names
input_data = {}
for feature in feature_names:
    input_data[feature] = st.text_input(f"Enter value for {feature}")

# Predict button
if st.button("Predict"):
    try:
        # Create a DataFrame with the correct feature order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Convert all input values to numeric
        input_df = input_df.apply(pd.to_numeric)

        # Make prediction
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Value: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

