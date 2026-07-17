import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write("🎯 **Production Safe Engine:** Fixed Initialization Vectors to Eliminate Trading Drift")

# =====================================================================
# MATHEMATICAL ENGINES (Causal State Machines)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
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

def calculate_rolling_hurst_leak_free(price_series, window=100):
    hurst_values = np.full(len(price_series), 0.5) 
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i - window + 1 : i + 1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        
        rs_ratio = r_val / s_val
        if rs_ratio > 0:
            h = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h, 0.0, 1.0)
            
    return hurst_values

# -----------------------------------------------------------------
# 🛡️ SYSTEM DATA INGESTION & COLD INITIATION
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching 2-Year Hourly Bitcoin Data from Yahoo Finance..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Live running candle protection locked
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        else:
            st.error("🚨 Error: Insufficient data from API.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# ⚡ GLOBAL CALCULATIONS (Warm-up System Matrix to Stop Drift)
# =====================================================================
# Warm-up phase runs globally across complete timeline data array
close_arr_global = df['Close'].values

# 1. Base Kalman System
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr_global, initial_p=50.0, q_val=0.0005, r_val=0.2)

# 2. Pure Hurst Base Calculation
df['Hurst'] = calculate_rolling_hurst_leak_free(close_arr_global, window=100)

# 3. Smooth Hurst Engine Layer using Kalman with P=0.50
df['Hurst_Kalman'] = apply_kalman_filter_custom(df['Hurst'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 4. Raw Price-based Weighted Momentum
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 5. Hurst Amplitude Weighted Momentum Core
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst_Kalman'] * 2.0)

# =====================================================================
# 🔥 PRODUCTION SAFE ISOLATION VECTOR (Strict 50:50 Display Allocation)
# =====================================================================
# Process mathematical lines first, THEN cut the frame in half.
# This preserves structural memory continuity for trading execution vectors.
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()
df_predict.dropna(subset=['Hurst'], inplace=True)

st.success(f"🟢 **Trading Engine Stabilized! {len(df_predict)} Candles Locked Without Historical Indicator Drift.**")

# =====================================================================
# 📋 MATRIX FORMATTING AND UTC DISPLAY
# =====================================================================
clean_cols = [
    'Close', 
    'Hurst',
    'Hurst_Kalman',
    'Weighted_Momentum', 
    'Hurst_Amp_Momentum'
]
display_df = df_predict[clean_cols].copy()

# Rename to match layout requirement specs cleanly
display_df.rename(columns={'Close': 'Close_Raw'}, inplace=True)

for c in display_df.columns:
    display_df[c] = display_df[c].round(2)

# Order framework with latest active matrix states on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M UTC')

st.subheader("📋 Production Bounded Kinematic Execution Matrix")
st.dataframe(display_df, use_container_width=True, height=650)
