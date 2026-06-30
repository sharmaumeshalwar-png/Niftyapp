import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

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
    
    x_hat = prices[0]  
    P = 1.0            
    
    for t in range(n_timestamps):
        x_hat_minus = x_hat
        P_minus = P + Q
        
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

start_date = datetime(2025, 1, 1, 0, 0)

if data_mode == "Generate 1-Hour Candle Data":
    # 1 Jan 2025 se June 2026 tak lagatar fixed safe length ka data loop
    total_hours = 5000  # Safe rendering limits for Streamlit dataframes
    
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    np.random.seed(42)
    steps = np.random.normal(0, 0.3, total_hours)
    close_prices = 150 + np.cumsum(steps)
    
    df = pd.DataFrame({"Timestamp": timestamps, "Close_A": close_prices})
    df.set_index("Timestamp", inplace=True)

else:
    uploaded_file = st.sidebar.file_uploader("CSV file upload", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
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
# Processing Framework 
# -------------------------------------------------------------------------
df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.1)

df['Feature_A_Lag'] = df['Close_A']
df['Feature_B_Lag'] = df['Kalman_B']
df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
df_clean = df.dropna()

split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

X_train = train_df[['Feature_A_Lag', 'Feature_B_Lag']]
y_train = train_df['Target_D_Next_Hour']
X_test = test_df[['Feature_A_Lag', 'Feature_B_Lag']]
y_test = test_df['Target_D_Next_Hour']

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # Optimised trees count
model.fit(X_train, y_train)

test_df = test_df.copy()
test_df['ML_Prediction_D'] = model.predict(X_test)

# -------------------------------------------------------------------------
# Streamlit Dashboard Output FIXED DATA VIEW
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Output Data Matrix Table (C & D Results)")

# Core Fix: Simple Table Conversion for Streamlit to prevent blank views
output_table = test_df[['Close_A', 'Kalman_B', 'Target_D_Next_Hour', 'ML_Prediction_D']].copy()
output_table.columns = [
    "Actual Close (A)", 
    "Kalman Smooth (B)", 
    "Next Hour Target (True D)", 
    "ML Predict Output (Predicted D)"
]

# Resetting index to show Date-Time cleanly as a visible column in the matrix
output_table = output_table.reset_index()

# Streamlit Native Table Component (Always forces data rendering without getting blank)
st.table(output_table.tail(15))
