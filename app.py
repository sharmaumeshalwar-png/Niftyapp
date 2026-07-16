import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC SMO + Kalman Hybrid", layout="wide")
st.title("🚀 Bitcoin (BTC-USD) SMO + Kalman Hybrid Engine")
st.write("🎯 **Dual-Engine Analytics:** SMO Rocket Speed smoothed with a strict 0.50 Kalman Filter | 100% Zero-Leakage")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal - SMO & New Kalman Filter)
# =====================================================================
def apply_sliding_mode_observer_damped(price_array, k_gain=0.25, l_gain=0.10, drag=0.88):
    """
    Damped Sliding Mode Observer (SMO)
    Calculates rocket trajectory velocity without leakage.
    """
    if len(price_array) == 0: return [], []
    x1 = price_array[0]
    x2 = 0.0
    est_price = []
    est_velocity = []
    
    for z in price_array:
        error = z - x1
        switching_control = k_gain * np.sign(error) if error != 0 else 0.0
        dx1 = x2 + (l_gain * error) + switching_control
        x1 = x1 + dx1
        
        dx2 = 0.05 * switching_control
        x2 = (x2 * drag) + dx2  
        
        est_price.append(x1)
        est_velocity.append(x2)
        
    return est_price, est_velocity

def apply_kalman_filter_custom(data_array, initial_p=0.50, q_val=0.005, r_val=0.005):
    """
    Custom Kalman Filter tailored specifically for stabilizing momentum vectors.
    Using balanced process noise (q) and measurement noise (r) ratio for a 0.50 baseline.
    """
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
            
            # ⛔ CUTOFF ENGINE: Drop the active unclosed 1-hour candle.
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

# 1. Apply SMO Trajectory 
est_p, est_v = apply_sliding_mode_observer_damped(close_arr, k_gain=0.25, l_gain=0.10, drag=0.88)
df['SMO_Price_Baseline'] = est_p
df['SMO_Velocity'] = est_v

# 2. Hurst Vector Generation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 3. Raw Hurst Amplified Momentum (SMO Velocity base)
df['Hurst_Amp_Momentum'] = df['SMO_Velocity'] * (df['Hurst'] * 100.0)

# Clean dynamic NaNs safely before mapping secondary Kalman
df.dropna(subset=['Hurst', 'Hurst_Amp_Momentum'], inplace=True)

# 🎯 4. NEW REQ: Apply 0.50 Kalman Filter directly onto Hurst_Amp_Momentum
# Custom-tuned parameter mapping representing 0.50 structural covariance scale
ham_raw_vals = df['Hurst_Amp_Momentum'].values
df['HAM_Kalman'] = apply_kalman_filter_custom(ham_raw_vals, initial_p=0.50, q_val=0.005, r_val=0.005)

# 5. Volume Rolling Mean for Causal Strength Analysis
df['Vol_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().ffill().fillna(0)

# 6. SMO + Kalman Volume-Momentum Decision Engine
# Now using HAM_Kalman instead of raw Hurst_Amp_Momentum to avoid choppy signals!
vol_mom_decisions = []
for i in range(len(df)):
    curr_kalman_amp = df['HAM_Kalman'].iloc[i]
    curr_vol = df['Volume'].iloc[i]
    avg_vol = df['Vol_MA_20'].iloc[i]
    
    if curr_kalman_amp > 0.10 and curr_vol > avg_vol:
        status = "🟢 HYBRID BULL"
    elif curr_kalman_amp < -0.10 and curr_vol > avg_vol:
        status = "🔴 HYBRID BEAR"
    else:
        status = "⚪ NEUTRAL FLAT"
    vol_mom_decisions.append(status)

df['Vol_Mom_Decision'] = vol_mom_decisions

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Strict Grid Alignment)
# =====================================================================
df_predict = df.copy()

st.success("🟢 **HAM Kalman 0.50 Activated:** Smoothed rocket speed momentum nested as a core decision metric!")

# Target Display Order
clean_cols = [
    'Close', 'SMO_Price_Baseline', 'SMO_Velocity', 'Volume', 
    'Hurst', 'Hurst_Amp_Momentum', 'HAM_Kalman', 'Vol_Mom_Decision'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding
display_df['Close_Raw'] = display_df['Close_Raw'].round(2)
display_df['SMO_Price_Baseline'] = display_df['SMO_Price_Baseline'].round(2)
display_df['SMO_Velocity'] = display_df['SMO_Velocity'].round(4)
display_df['Volume'] = display_df['Volume'].round(0)
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)
display_df['HAM_Kalman'] = display_df['HAM_Kalman'].round(4) # New Column Rounded

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Hour Hybrid SMO-Kalman Board")
st.dataframe(display_df, use_container_width=True, height=750)
