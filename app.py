import streamlit as st
import joblib
import os
import numpy as np  # FIX: Added missing import

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the paths to your files correctly
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ---------------------------
# Load Model & Scaler with Error Handling
# ---------------------------
@st.cache_resource # Optimization: only load these once
def load_assets():
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or Scaler files not found! Please run your training script first.")
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_assets()

# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")
st.title("📱 Mobile Price Prediction App")
st.markdown("Enter the specifications below to estimate the price category.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", value=2000, step=100)
    ram = st.number_input("RAM (MB)", value=2048, step=128)
    px_height = st.number_input("Pixel Height", value=1000, step=10)

with col2:
    px_width = st.number_input("Pixel Width", value=1000, step=10)
    mobile_wt = st.number_input("Mobile Weight (grams)", value=150, step=5)
    int_memory = st.number_input("Internal Memory (GB)", value=32, step=4)

# ---------------------------
# Prediction Logic
# ---------------------------
if st.button("Predict Price", type="primary"):
    if model is not None and scaler is not None:
        try:
            # 1. Create the array with the 6 features
            # Ensure the order matches exactly: [battery_power, ram, px_height, px_width, mobile_wt, int_memory]
            input_data = np.array([[battery_power, ram, px_height, px_width, mobile_wt, int_memory]])
            
            # 2. Scale the data
            input_scaled = scaler.transform(input_data)
            
            # 3. Make prediction
            prediction = model.predict(input_scaled)
            
            # 4. Map and Show result
            price_dict = {
                0: "Low Price 💰", 
                1: "Medium Price 💰💰", 
                2: "High Price 💰💰💰", 
                3: "Very High Price 💎"
            }
            
            result = price_dict.get(prediction[0], "Unknown Range")
            st.success(f"### Predicted Category: {result}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Prediction unavailable because model files are missing.")