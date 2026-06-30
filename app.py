import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Page Configuration
st.set_page_config(page_title="Kalman + ML Matrix Predictor", layout="wide")
st.title("📊 Kalman Filter & Machine Learning Hybrid Matrix Table")
st.write("A = Close Price | B = Kalman Filter (Q=0.0001) | C = Features Combination | D = ML Target Prediction")
st.write("**Data Structure:** 1-Hour Candles | **Start Date:** 01 January 2025")

# -------------------------------------------------------------------------
# Helper Function: 1D Kalman Filter Implementation
# -------------------------------------------------------------------------
def apply_kalman_filter(prices, Q=0.0001, R=0.1):
    n_timestamps = len(prices)
    filtered_prices = np.zeros(n_timestamps)
    
    # Initial guesses
    x_hat = prices[0]  
    P = 1.0            
    
    for t in range(n_timestamps):
        # 1. Prediction Step
        x_hat_minus = x_hat
        P_minus = P + Q
        
        # 2. Measurement Update Step (Correction)
        K = P_minus / (P_minus + R)  
        x_hat = x_hat_minus + K * (prices[t] - x_hat_minus)
        P = (1 - K) * P_minus
        
        filtered_prices[t] = x_hat
        
    return filtered_prices

# -------------------------------------------------------------------------
# Data Setup: 1-Hour Candles starting from 1st Jan 2025
# -------------------------------------------------------------------------
st.sidebar.header("🗓️ Data Filter Settings")
data_mode = st.sidebar.selectbox("Data Source", ["Generate 1-Hour Candle Data", "Upload Hourly CSV"])

# Current system time (June 2026) tak ka range generate karne ke liye
start_date = datetime(2025, 1, 1, 0, 0)

if data_mode == "Generate 1-Hour Candle Data":
    # Calculating total hours from Jan 1, 2025 to June 2026 (approx 13000 hours)
    total_hours = 13000 
    
    # Generate Timestamp Index
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    # Generate Synthetic Random Walk for Hourly Close Prices (A)
    np.random.seed(42)
    steps = np.random.normal(0, 0.3, total_hours)
    close_prices = 150 + np.cumsum(steps)
    
    df = pd.DataFrame({"Timestamp": timestamps, "Close_A": close_prices})
    df.set_index("Timestamp", inplace=True)

else:
    uploaded_file = st.sidebar.file_uploader("CSV file upload karein (Isme 'Timestamp' aur 'Close' hona chahiye)", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        # Convert timestamp and filter from 1st Jan 2025
        if 'Timestamp' in user_df.columns and 'Close' in user_df.columns:
            user_df['Timestamp'] = pd.to_datetime(user_df['Timestamp'])
            user_df = user_df[user_df['Timestamp'] >= pd.to_datetime('2025-01-01')]
            
            df = pd.DataFrame({
                "Timestamp": user_df['Timestamp'].values,
                "Close_A": user_df['Close'].values
            })
            df.set_index("Timestamp", inplace=True)
        else:
            st.error("CSV mein 'Timestamp' aur 'Close' columns zaroori hain!")
            st.stop()
    else:
        st.info("Hourly CSV upload karein ya fir default 'Generate' option chunein.")
        st.stop()

# -------------------------------------------------------------------------
# Processing Framework (8-Step Logic Implementation)
# -------------------------------------------------------------------------

# Step 1 & 2: Calculate Kalman Filter (B) on Close Price (A)
df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.1)

# Step 3: Feature Matrix C Creation (Using lag inputs)
df['Feature_A_Lag'] = df['Close_A']
df['Feature_B_Lag'] = df['Kalman_B']

# Target Variable D: Next Hour's Close Price
df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
df_clean = df.dropna()

# Step 4 & 5: Train-Test Split (80% Train / 20% Test)
split_idx = int
