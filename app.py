import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Z-Score Precision Engine", layout="wide")
st.title("⚡ Z-Score Normalized Value Master Engine")
st.write("🎯 **Pure Value Focus:** Hurst-Amplified Momentum with Rolling Z-Score Standardization (100% Leakage-Free & Uniform Scaling)")

# =====================================================================
# MATHEMATICAL ENGINES (Zero Lag Core Math)
# =====================================================================
def apply_kalman_filter_precise(data_array, initial_p=50.0, q_val=0.005, r_val=0.05):
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
target_ticker = st.selectbox("Select Asset to Generate Core Value Matrix:", ["BTC-USD", "ETH-USD", "SI=F", "RELIANCE.NS"])

df = None

with st.spinner(f"Processing Data Streams for {target_ticker}..."):
    try:
        df = yf.download(tickers=target_ticker, period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Strict Leakage Protection Drop
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data lines.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# 🔥 CORE MATHEMATICAL ENGINE PIPELINE
# =====================================================================
close_arr = df['Close'].values

# Baseline generation
df['Kalman_Baseline'] = apply_kalman_filter_precise(close_arr, initial_p=50.0, q_val=0.01, r_val=0.02)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Momentum derivation
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_precise(raw_weighted_momentum.values, initial_p=0.50, q_val=0.005, r_val=0.05)

# THE EXPONENTIAL MULTIPLICATION (Raw Indicator Value)
df['Hurst_Amp_Momentum_Raw'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# 🎯 THE MATHEMAGIC: ROLLING Z-SCORE NORMALIZATION (Pure Scaling Layer)
# Yeh bina kisi lookahead bias ke pichle 50 bars ki memory ke basis par scale reset karta hai
rolling_window = 50
rolling_mean = df['Hurst_Amp_Momentum_Raw'].rolling(window=rolling_window, min_periods=1).mean()
rolling_std = df['Hurst_Amp_Momentum_Raw'].rolling(window=rolling_window, min_periods=1).std().fillna(1.0)

# Formula: (Current_Value - Mean) / Standard_Deviation
df['Z_Score_Normalized_Value'] = (df['Hurst_Amp_Momentum_Raw'] - rolling_mean) / (rolling_std + 1e-8)

df.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 🎛️ STRICT LEARNING FILTER (1-Year Slicing Layer)
# =====================================================================
cutoff_date = df.index.max() - timedelta(days=365)
df_predict = df[df.index >= cutoff_date].copy()

st.success(f"🟢 **Z-Score Normalization Engine Fully Armed:** Ineffective ATR filters removed completely.")

# Clean matrix containing only the numeric parameters you care about
clean_cols = ['Close', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'Z_Score_Normalized_Value']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Raw_Price'}, inplace=True)

# Precision rounding for crystal clear visibility
for c in ['Raw_Price', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'Z_Score_Normalized_Value']:
    display_df[c] = display_df[c].round(4)

# Chronological sorting for table reading
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Amplified Z-Score Matrix (Track 'Z_Score_Normalized_Value' for clean standard swings)")
st.dataframe(display_df, use_container_width=True, height=750)
