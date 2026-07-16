import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Sliding Mode Observer Engine", layout="wide")
st.title("🚀 Bitcoin (BTC-USD) Sliding Mode Observer Engine")
st.write("🎯 **Aerospace Grade Trading:** 1-Hour Candles | Sliding Mode Observer (SMO) | 100% Zero-Leakage")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal - Sliding Mode Observer Theory)
# =====================================================================
def apply_sliding_mode_observer(price_array, k_gain=0.15, l_gain=0.05):
    """
    Sliding Mode Observer (SMO) for State Estimation (Speed/Velocity)
    Mathematically proven rocket trajectory tracking without Kalman Filters.
    """
    if len(price_array) == 0: return [], []
    
    # State variables: x1 (Estimated Price), x2 (Estimated Speed/Velocity)
    x1 = price_array[0]
    x2 = 0.0
    
    est_price = []
    est_velocity = []
    
    for z in price_array:
        # 1. Calculate Observer Error (Output Error)
        error = z - x1
        
        # 2. Sliding Surface Switching Logic (Signum Multiplier)
        switching_control = k_gain * np.sign(error) if error != 0 else 0.0
        
        # 3. State Update Equations (Discontinuous Correction)
        # dx1/dt = x2 + L * error + K * sign(error)
        dx1 = x2 + (l_gain * error) + switching_control
        x1 = x1 + dx1  # Update virtual position
        
        # dx2/dt = K_speed * sign(error) -> Integrated velocity update
        dx2 = 0.01 * switching_control
        x2 = x2 + dx2  # Update virtual speed
        
        est_price.append(x1)
        est_velocity.append(x2)
        
    return est_price, est_velocity

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

# 1. APPLY SLIDING MODE OBSERVER (Replacing Kalman Filter completely)
# k_gain=0.15 represents the switching intensity, l_gain=0.05 is the error feedback
est_p, est_v = apply_sliding_mode_observer(close_arr, k_gain=0.15, l_gain=0.05)
df['SMO_Price_Baseline'] = est_p
df['SMO_Velocity'] = est_v

# 2. Hurst Vector Generation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 3. Hurst Amplified Momentum Pipeline (Using pure SMO output instead of Kalman)
df['Hurst_Amp_Momentum'] = df['SMO_Velocity'] * (df['Hurst'] * 2000.0) # Scaling velocity for readability

# Clean dynamic NaNs
df.dropna(subset=['Hurst', 'Hurst_Amp_Momentum'], inplace=True)

# 4. Volume Rolling Mean for Causal Strength Analysis
df['Vol_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().ffill().fillna(0)

# 5. SMO Volume-Momentum Decision Engine
vol_mom_decisions = []
for i in range(len(df)):
    curr_amp = df['Hurst_Amp_Momentum'].iloc[i]
    curr_vol = df['Volume'].iloc[i]
    avg_vol = df['Vol_MA_20'].iloc[i]
    
    if curr_amp > 0.05 and curr_vol > avg_vol:
        status = "🟢 SMO BULL"
    elif curr_amp < -0.05 and curr_vol > avg_vol:
        status = "🔴 SMO BEAR"
    else:
        status = "⚪ NEUTRAL FLAT"
    vol_mom_decisions.append(status)

df['Vol_Mom_Decision'] = vol_mom_decisions

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Strict Columns Formatting Layer)
# =====================================================================
df_predict = df.copy()

st.success("🟢 **Sliding Mode Observer Locked:** Kalman Filter completely purged. Trajectory-tracking physics applied.")

# Custom Rocket Matrix Layout
clean_cols = [
    'Close', 'SMO_Price_Baseline', 'SMO_Velocity', 'Volume', 
    'Hurst', 'Hurst_Amp_Momentum', 'Vol_Mom_Decision'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding Layers
display_df['Close_Raw'] = display_df['Close_Raw'].round(2)
display_df['SMO_Price_Baseline'] = display_df['SMO_Price_Baseline'].round(2)
display_df['SMO_Velocity'] = display_df['SMO_Velocity'].round(6) # Highly precise trajectory feedback
display_df['Volume'] = display_df['Volume'].round(0)
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Hour Rocket SMO Analytics Board")
st.dataframe(display_df, use_container_width=True, height=750)
