import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Direct Crypto Signals:** 50:50 Split 1-Hour Candle Leak-Free Execution Matrix")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Backward-Looking / No Leakage)
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
    
    # 🛡️ LEAKAGE PROTECTION STEP: No np.roll boundary wrapping. Absolute backward shift.
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    
    for i in range(window, len(price_series)):
        # Slices from (i - window + 1) up to and including current index i
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
# 🛡️ SYSTEM DATA INGESTION (BTC-USD Setup - 2y, 1h)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching 2-Year Hourly Bitcoin Data from Yahoo Finance..."):
    try:
        # Strictly 2 Years data at 1-Hour intervals
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Live Running Candle Protection (Strictly No Leakage)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        else:
            st.error("🚨 Error: Insufficient data lines from Yahoo Finance.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# 🔥 CRITICAL DATA SCIENCE RULE: Strict 50:50 split FIRST to secure forward vectors
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Bitcoin Candles (Leak-Free Engine Active)!**")

# =====================================================================
# ⚡ CORE PURE KINEMATICS MATRIX CALCULATIONS
# =====================================================================
df_predict['Close_Raw'] = df_predict['Close']
close_arr = df_predict['Close_Raw'].values

# 1. Base Kalman System for Baseline Extraction
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# 2. Pure Hurst Base Calculation (Leak-Proof Windowing)
df_predict['Hurst'] = calculate_rolling_hurst_leak_free(close_arr, window=100)

# 3. Raw Price-based Weighted Momentum
raw_weighted_momentum = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 4. Hurst Amplitude Weighted Momentum Core
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

# Clean NaNs strictly before UI pipeline formatting
df_predict.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 📋 MATRIX FORMATTING AND UTC DISPLAY
# =====================================================================
clean_cols = [
    'Close_Raw', 
    'Hurst', 
    'Weighted_Momentum', 
    'Hurst_Amp_Momentum'
]
display_df = df_predict[clean_cols].copy()

# Precision setting to 2 decimal points for mathematical readability
for c in clean_cols:
    display_df[c] = display_df[c].round(2)

# Latest candles on top display rule for active traders
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M UTC')

st.subheader("📋 50:50 Split Pure Kinematic Analysis Matrix")
st.dataframe(display_df, use_container_width=True, height=650)
