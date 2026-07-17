import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Pure Newton Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Newtonian Suite:** Bounded Equations of Motion Vector Layer ($v, a, s$) without Hurst Matrix")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Backward-Looking & Continuous)
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
close_arr_global = df['Close'].values

# 1. Base Price Kalman System (The Frame of Reference)
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr_global, initial_p=50.0, q_val=0.0005, r_val=0.2)

# 2. Pure Kinetic Velocity (v) Matrix
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']
df['Newton_Velocity'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# =====================================================================
# 🏎️ NEWTON PURE KINETIC SUITE EQUATIONS (No Hurst Dependency)
# =====================================================================
# Initial Velocity (u) is the shifted state of velocity vector (t-1)
df['Newton_Initial_Velocity'] = df['Newton_Velocity'].shift(1).fillna(0.0)

# Market mass based on standard deviation volatility matrix
market_mass = df['Close'].rolling(window=20).std().fillna(1.0)
df['Velocity_Delta'] = df['Newton_Velocity'] - df['Newton_Initial_Velocity']

# Equation 1: Acceleration (a = F / m)
df['Newton_Acceleration'] = (df['Velocity_Delta'] / (market_mass + 1e-5)) * 100.0

# Equation 2: Displacement Vector -> s = ut + 0.5 * a * t^2 (Assuming delta t = 1 candle unit)
df['Newton_Displacement'] = df['Newton_Initial_Velocity'] + (0.5 * df['Newton_Acceleration'])

# =====================================================================
# 🔥 PRODUCTION SAFE ISOLATION VECTOR (Strict 50:50 Display Allocation)
# =====================================================================
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Pure Newtonian Physics Engine Active! Hurst Layer completely dismantled.**")

# =====================================================================
# 📋 MATRIX FORMATTING AND UTC DISPLAY
# =====================================================================
clean_cols = [
    'Close', 
    'Newton_Velocity', 
    'Newton_Acceleration',
    'Newton_Displacement'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw'}, inplace=True)

for c in display_df.columns:
    if c in ['Newton_Acceleration', 'Newton_Displacement']:
        display_df[c] = display_df[c].round(4)
    else:
        display_df[c] = display_df[c].round(2)

# Order framework with latest active matrix states on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M UTC')

st.subheader("📋 Production Bounded Pure Newtonian Kinematic Matrix")
st.dataframe(display_df, use_container_width=True, height=650)
