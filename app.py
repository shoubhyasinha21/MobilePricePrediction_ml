import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("📱 Mobile Price Prediction App")

# Input Fields
battery_power = st.number_input("Battery Power (mAh)", value=2000)
ram = st.number_input("RAM (MB)", value=2048)
px_height = st.number_input("Pixel Height", value=1000)
px_width = st.number_input("Pixel Width", value=1000)
mobile_wt = st.number_input("Mobile Weight (grams)", value=150)
int_memory = st.number_input("Internal Memory (GB)", value=32)

if st.button("Predict Price"):
    # Just create an array with the 6 features in the SAME order as 'features' list above
    input_data = np.array([[battery_power, ram, px_height, px_width, mobile_wt, int_memory]])
    
    # This will now work without errors because the scaler expects 6 features
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    price_dict = {0: "Low Price", 1: "Medium Price", 2: "High Price", 3: "Very High Price"}
    st.success(f"Predicted Price Range: {price_dict[prediction[0]]}")