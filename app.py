import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Kalman Gap Engine", layout="wide")
st.title("⚡ Kalman Convergence/Divergence Gap Engine")
st.write("🎯 **Pure Value Edge:** Dual-Speed Kalman Tracking Array applied on Hurst-Momentum Matrix (100% Leakage-Free)")

# =====================================================================
# MATHEMATICAL ENGINES (Zero Lag Core Math Architecture)
# =====================================================================
def apply_kalman_filter_dynamic(data_array, initial_p=50.0, q_val=0.005, r_val=0.05):
    """Rigid state tracking loop with customizable parameters"""
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
# 🛡️ SYSTEM DATA INGESTION (Target Selection Block)
# -----------------------------------------------------------------
target_ticker = st.selectbox("Select Target Matrix Asset:", ["BTC-USD", "ETH-USD", "SI=F", "RELIANCE.NS"])

df = None

with st.spinner(f"Processing Data Streams for {target_ticker}..."):
    try:
        df = yf.download(tickers=target_ticker, period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Strict Leakage Protection Lock (Patthar ki Lakeer)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient historical data lines.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 Ingestion Rail Failure: {e}")
        st.stop()

# =====================================================================
# 🔥 DUAL-KALMAN CONVERGENCE PIPELINE
# =====================================================================
close_arr = df['Close'].values

# Base Step 1: Ingest Price Baseline
df['Price_Baseline'] = apply_kalman_filter_dynamic(close_arr, initial_p=50.0, q_val=0.01, r_val=0.02)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Base Step 2: Derive Raw Weighted Momentum
raw_weighted_momentum = df['Close'] - df['Price_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_dynamic(raw_weighted_momentum.values, initial_p=0.50, q_val=0.005, r_val=0.05)

# 🎯 THE SUPREME RAW VALUE MATRIX
df['Hurst_Amp_Momentum_Raw'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)
raw_ham_values = df['Hurst_Amp_Momentum_Raw'].values

# --- 🧠 PAJI'S MAGICAL DUAL-KALMAN GAUNTLET IMPLEMENTATION ---

# ⚡ Layer A: Fast Kalman Tracking (High agility to capture immediate shifts)
# q_val bada aur r_val chota rakhne se yeh bina delay ke momentum ko chipak kar chalta hai
df['Fast_Kalman_HAM'] = apply_kalman_filter_dynamic(raw_ham_values, initial_p=1.0, q_val=0.02, r_val=0.01)

# 🐢 Layer B: Slow Kalman Tracking (High damping factor to form the solid trend line)
# q_val chota aur r_val bada rakhne se yeh market ki noise ko filter out karke chalta hai
df['Slow_Kalman_HAM'] = apply_kalman_filter_dynamic(raw_ham_values, initial_p=1.0, q_val=0.002, r_val=0.15)

# 🏁 Layer C: THE DUAL GAP CONVERGENCE (Fast - Slow)
df['KALMAN_GAP_VALUE'] = df['Fast_Kalman_HAM'] - df['Slow_Kalman_HAM']

df.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 🎛️ STRICT LEARNING FILTER (1-Year Slicing Array)
# =====================================================================
cutoff_date = df.index.max() - timedelta(days=365)
df_predict = df[df.index >= cutoff_date].copy()

st.success(f"🟢 **Dual-Kalman Convergence Grid Loaded:** Numerical gap extraction operational for {target_ticker}!")

# Purely numeric parameters focused matrix
clean_cols = ['Close', 'Hurst_Amp_Momentum_Raw', 'Fast_Kalman_HAM', 'Slow_Kalman_HAM', 'KALMAN_GAP_VALUE']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Raw_Price'}, inplace=True)

# Formatting for precise numerical visibility
for c in ['Raw_Price', 'Hurst_Amp_Momentum_Raw', 'Fast_Kalman_HAM', 'Slow_Kalman_HAM', 'KALMAN_GAP_VALUE']:
    display_df[c] = display_df[c].round(4)

# Reverse chronological sorting for intuitive grid tracking
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Pure Dual-Kalman Execution Grid (Monitor the contraction/expansion of 'KALMAN_GAP_VALUE')")
st.dataframe(display_df, use_container_width=True, height=750)
