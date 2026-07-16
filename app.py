import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Damped SMO Engine", layout="wide")
st.title("🚀 Bitcoin (BTC-USD) Damped Sliding Mode Observer")
st.write("🎯 **Aerospace Grade Trading:** 1-Hour Candles | SMO with Atmospheric Drag | 100% Zero-Leakage")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal - SMO with Atmospheric Drag)
# =====================================================================
def apply_sliding_mode_observer_damped(price_array, k_gain=0.15, l_gain=0.05, drag=0.90):
    """
    Damped Sliding Mode Observer (SMO)
    Adds physical 'atmospheric drag' to stop infinite speed drift/leakage.
    """
    if len(price_array) == 0: return [], []
    
    # State variables: x1 (Estimated Price), x2 (Estimated Speed)
    x1 = price_array[0]
    x2 = 0.0
    
    est_price = []
    est_velocity = []
    
    for z in price_array:
        # 1. Output error
        error = z - x1
        
        # 2. Binary switching logic
        switching_control = k_gain * np.sign(error) if error != 0 else 0.0
        
        # 3. Trajectory Update
        dx1 = x2 + (l_gain * error) + switching_control
        x1 = x1 + dx1
        
        # 4. Damped Velocity Update (Applying 0.90 Drag factor to prevent infinite minus drift)
        dx2 = 0.05 * switching_control
        x2 = (x2 * drag) + dx2  # Drag pulls velocity back to zero when trend slows down
        
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
            
            # ⛔ CUTOFF ENGINE: Drop the active unclosed 1-hour candle
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

# 1. APPLY DAMPED SMO (Atmospheric friction locked)
# k_gain=0.25, l_gain=0.10 for fast tracking, drag=0.88 for strong stabilization
est_p, est_v = apply_sliding_mode_observer_damped(close_arr, k_gain=0.25, l_gain=0.10, drag=0.88)
df['SMO_Price_Baseline'] = est_p
df['SMO_Velocity'] = est_v

# 2. Hurst Vector Generation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 3. Hurst Amplified Momentum Pipeline (Scaling with damped velocity)
df['Hurst_Amp_Momentum'] = df['SMO_Velocity'] * (df['Hurst'] * 100.0)

# Clean dynamic NaNs
df.dropna(subset=['Hurst', 'Hurst_Amp_Momentum'], inplace=True)

# 4. Volume Rolling Mean
df['Vol_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().ffill().fillna(0)

# 5. SMO Volume-Momentum Decision Engine (Slightly adjusted thresholds)
vol_mom_decisions = []
for i in range(len(df)):
    curr_amp = df['Hurst_Amp_Momentum'].iloc[i]
    curr_vol = df['Volume'].iloc[i]
    avg_vol = df['Vol_MA_20'].iloc[i]
    
    if curr_amp > 0.10 and curr_vol > avg_vol:
        status = "🟢 SMO BULL"
    elif curr_amp < -0.10 and curr_vol > avg_vol:
        status = "🔴 SMO BEAR"
    else:
        status = "⚪ NEUTRAL FLAT"
    vol_mom_decisions.append(status)

df['Vol_Mom_Decision'] = vol_mom_decisions

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Strict Columns Formatting Layer)
# =====================================================================
df_predict = df.copy()

st.success("🟢 **SMO Friction Calibrated:** Atmospheric drag applied. Speed will now naturally revert to zero during flat markets!")

# Display order
clean_cols = [
    'Close', 'SMO_Price_Baseline', 'SMO_Velocity', 'Volume', 
    'Hurst', 'Hurst_Amp_Momentum', 'Vol_Mom_Decision'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding
display_df['Close_Raw'] = display_df['Close_Raw'].round(2)
display_df['SMO_Price_Baseline'] = display_df['SMO_Price_Baseline'].round(2)
display_df['SMO_Velocity'] = display_df['SMO_Velocity'].round(4) # Clean responsive speed
display_df['Volume'] = display_df['Volume'].round(0)
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Hour Rocket SMO Analytics Board")
st.dataframe(display_df, use_container_width=True, height=750)
