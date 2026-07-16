import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Dual-Kalman Engine", layout="wide")
st.title("⚡ Dual-Stage Kalman Momentum Engine")
st.write("🎯 **Pure State Estimation:** Raw Hurst-Momentum passed through a secondary Hyper-Sensitive Kalman Filter (100% Leakage-Free)")

# =====================================================================
# MATHEMATICAL ENGINES (Zero Lag Core Math)
# =====================================================================
def apply_kalman_filter_precise(data_array, initial_p=50.0, q_val=0.005, r_val=0.05):
    """Core Kalman Filter for state estimation"""
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p  
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def calculate_rolling_hurst(price_series, window=100):
    hurst_values = np.full(len(price_series), 0.5) 
    log_returns = np.log(price_series / np.roll(price_series, 1))
    log_returns[0] = 0
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i-window+1:i+1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        rs_ratio = r_val / s_val
        h = np.log(rs_ratio) / np.log(window)
        hurst_values[i] = np.clip(h, 0.0, 1.0)
    return hurst_values

# -----------------------------------------------------------------
# 🛡️ SYSTEM DATA INGESTION (Target Selection)
# -----------------------------------------------------------------
target_ticker = st.selectbox("Select Target Asset:", ["BTC-USD", "ETH-USD", "SI=F", "RELIANCE.NS"])

df = None

with st.spinner(f"Processing Data Streams for {target_ticker}..."):
    try:
        df = yf.download(tickers=target_ticker, period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Patthar ki Lakeer: Strict Leakage Protection Drop
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data lines.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 Ingestion Failure: {e}")
        st.stop()

# =====================================================================
# 🔥 DUAL-STAGE KALMAN PIPELINE
# =====================================================================
close_arr = df['Close'].values

# STAGE 1: Price Tracking Baseline (Fast Adaptive)
df['Price_Kalman_Baseline'] = apply_kalman_filter_precise(close_arr, initial_p=50.0, q_val=0.01, r_val=0.02)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Derive Raw Momentum
raw_weighted_momentum = df['Close'] - df['Price_Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_precise(raw_weighted_momentum.values, initial_p=0.50, q_val=0.005, r_val=0.05)

# Calculate Core Hurst Amplified Momentum
df['Hurst_Amp_Momentum_Raw'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# STAGE 2: THE MAGICAL FIX (Secondary Kalman applied directly on the Hurst-Momentum Value)
# Paji yahan humne hyper-sensitive parameters lagaye hain taaki smooth wave bane bina lag ke
raw_ham_values = df['Hurst_Amp_Momentum_Raw'].values
df['KALMAN_SMOOTHED_HAM'] = apply_kalman_filter_precise(raw_ham_values, initial_p=1.0, q_val=0.008, r_val=0.03)

df.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 🎛️ STRICT LEARNING FILTER (1-Year Slicing Layer)
# =====================================================================
cutoff_date = df.index.max() - timedelta(days=365)
df_predict = df[df.index >= cutoff_date].copy()

st.success(f"🟢 **Dual-Stage Kalman Filter Armed:** Value smoothed with absolute mathematical flow!")

# Clean matrix focused purely on the raw and smoothed value comparison
clean_cols = ['Close', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'KALMAN_SMOOTHED_HAM']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Raw_Price'}, inplace=True)

# Precision rounding for crystal clear reading
for c in ['Raw_Price', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'KALMAN_SMOOTHED_HAM']:
    display_df[c] = display_df[c].round(4)

# Chronological sorting
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Pure Value Matrix (Compare 'Hurst_Amp_Momentum_Raw' with 'KALMAN_SMOOTHED_HAM' to see the magic)")
st.dataframe(display_df, use_container_width=True, height=750)
