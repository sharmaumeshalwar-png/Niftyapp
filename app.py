import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Strict Core Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Core Decision Engine")
st.write("🎯 **Pure Vector Analytics:** 1-Hour Candles | Volume-Momentum Decision Framework | 100% Zero-Leakage Locked")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal - Forward Only)
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
# 🛡️ SYSTEM DATA INGESTION & RUNNING CANDLE PURGE
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live 1-Year BTC Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="1y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
            # ⛔ CUTOFF ENGINE: Drop the active unclosed dynamic 1-hour candle.
            df = df.iloc[:-1]
            
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
# 🔥 GLOBAL CALCULATION PIPELINE (Forward Flow Only)
# =====================================================================
close_arr = df['Close'].values

# 1. Price Kalman Baseline Calculation
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# 2. Dynamic Velocity Vector (Strictly Causal Close - Kalman)
df['Velocity'] = df['Close'] - df['Kalman_Baseline']

# 3. Hurst Vector Generation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 4. Hurst Amplified Momentum Pipeline
df['Close_Minus_Kalman'] = df['Close'] - df['Kalman_Baseline']
raw_weighted_momentum = df['Close_Minus_Kalman'].values
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum, initial_p=0.50, q_val=0.001, r_val=0.1)
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# Clean dynamic NaNs safely before matrix logical mapping
df.dropna(subset=['Hurst', 'Hurst_Amp_Momentum'], inplace=True)

# 5. Volume Rolling Mean for Causal Strength Analysis
df['Vol_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().ffill().fillna(0)

# 🎯 NEW DECISION ENGINE: Derived directly via Volume & Hurst Amplified Momentum
vol_mom_decisions = []
for i in range(len(df)):
    curr_amp = df['Hurst_Amp_Momentum'].iloc[i]
    curr_vol = df['Volume'].iloc[i]
    avg_vol = df['Vol_MA_20'].iloc[i]
    
    if curr_amp > 0.15 and curr_vol > avg_vol:
        status = "🟢 BULL STRENGTH"
    elif curr_amp < -0.15 and curr_vol > avg_vol:
        status = "🔴 BEAR PRESSURE"
    else:
        status = "⚪ NEUTRAL FLAT"
    vol_mom_decisions.append(status)

df['Vol_Mom_Decision'] = vol_mom_decisions

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Strict Columns Formatting Layer)
# =====================================================================
df_predict = df.copy()

st.success("🟢 **Core Matrix Re-engineered:** Clean dataset structured strictly around your chosen vectors.")

# Target structural order requested by the user
clean_cols = [
    'Close', 'Kalman_Baseline', 'Velocity', 'Volume', 
    'Hurst', 'Hurst_Amp_Momentum', 'Vol_Mom_Decision'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding Layers
display_df['Close_Raw'] = display_df['Close_Raw'].round(2)
display_df['Kalman_Baseline'] = display_df['Kalman_Baseline'].round(2)
display_df['Velocity'] = display_df['Velocity'].round(2)
display_df['Volume'] = display_df['Volume'].round(0) # Actual absolute Volume display count
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)

# Chronological sorting for visualization (latest closed bar on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Hour Purged Core Analytics Board")
st.dataframe(display_df, use_container_width=True, height=750)
