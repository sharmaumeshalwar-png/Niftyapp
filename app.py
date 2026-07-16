import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Amplified Value Engine", layout="wide")
st.title("⚡ Amplified Numeric Value Master Engine")
st.write("🎯 **Pure Value Focus:** Hurst-Amplified Momentum with Multiplier Scaling for High-Visibility Trading (100% Leakage-Free)")

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
            df = df.iloc[:-1] # Strict Leakage Protection Drop (Patthar ki Lakeer)
            
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

# True Range calculation without future bias
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill()

# Momentum derivation
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_precise(raw_weighted_momentum.values, initial_p=0.50, q_val=0.005, r_val=0.05)

# THE EXPONENTIAL MULTIPLICATION
df['Hurst_Amp_Momentum_Raw'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# 🎯 THE HIGH-VISIBILITY AMPLIFIED LAYER
# Paji yahan pehle normalise kiya ATR se, phir 100.0 se multiply kiya taaki sankhya badi dikhe!
df['Amplified_Momentum_Value'] = (df['Hurst_Amp_Momentum_Raw'] / (df['ATR'] + 1e-8)) * 100.0

df.dropna(subset=['ATR', 'Hurst'], inplace=True)

# =====================================================================
# 🎛️ STRICT LEARNING FILTER (1-Year Slicing Layer)
# =====================================================================
cutoff_date = df.index.max() - timedelta(days=365)
df_predict = df[df.index >= cutoff_date].copy()

st.success(f"🟢 **Visibility Engine Active:** Numbers successfully amplified by 100x factor!")

# Clean matrix containing only the numeric parameters you care about
clean_cols = ['Close', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'Amplified_Momentum_Value']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Raw_Price'}, inplace=True)

# Precision rounding
for c in ['Raw_Price', 'Hurst', 'Weighted_Momentum', 'Hurst_Amp_Momentum_Raw', 'Amplified_Momentum_Value']:
    display_df[c] = display_df[c].round(4)

# Chronological sorting for table reading
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Amplified Value Matrix (Check 'Amplified_Momentum_Value' for clear trend shifts)")
st.dataframe(display_df, use_container_width=True, height=750)
