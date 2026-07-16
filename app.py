import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Anti-Leakage Kinematics", layout="wide")
st.title("🚀 Bitcoin (BTC-USD) Newtonian Kinematics Engine")
st.write("🎯 **Physics-Based Analytics:** 1-Hour Candles | 100% Guaranteed Anti-Leakage & No Wrap-Around")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal - No Lookahead)
# =====================================================================
def calculate_kinematic_states(price_series, window=14):
    """
    Fits a local quadratic polynomial (y = A*t^2 + B*t + C) at each point 
    to calculate instantaneous velocity (dy/dt). Purely causal.
    """
    n = len(price_series)
    velocity = np.zeros(n)
    acceleration = np.zeros(n)
    
    t = np.arange(window)
    X = np.vstack([t**2, t, np.ones(window)]).T
    X_pseudo = np.linalg.pinv(X)
    
    for i in range(window - 1, n):
        # Slice only up to current index i (causal window)
        y = price_series[i - window + 1 : i + 1]
        beta = X_pseudo.dot(y)
        curr_t = window - 1
        
        # Velocity v = 2A*t + B
        velocity[i] = (2.0 * beta[0] * curr_t) + beta[1]
        # Acceleration a = 2A
        acceleration[i] = 2.0 * beta[0]
        
    return velocity, acceleration

def calculate_rolling_hurst_safe(price_series, window=100):
    """
    Safe Hurst calculation without np.roll wrap-around leakage.
    """
    hurst_values = np.full(len(price_series), 0.5)
    
    # 🛡️ STRICT ANTI-LEAK SHIFT (Zero wrap-around)
    # log_returns[t] = log(p[t] / p[t-1])
    log_returns = np.zeros(len(price_series))
    log_returns[1:] = np.log(price_series[1:] / price_series[:-1])
    
    for i in range(window, len(price_series)):
        # Strictly slice past windows, index 0 is safe/zeroed
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
            
            # ⛔ CUTOFF ENGINE: Immediately drop active unclosed candle
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
# 🔥 GLOBAL CALCULATION PIPELINE (Strict Forward Flow)
# =====================================================================
close_arr = df['Close'].values

# 1. Kinematics Physics Engine (No SMO/Kalman noise)
velocity, acceleration = calculate_kinematic_states(close_arr, window=14)
df['Kinematic_Velocity'] = velocity
df['Kinematic_Acceleration'] = acceleration

# 2. 🛡️ Safe Hurst Vector Generation (Leakage completely killed)
df['Hurst'] = calculate_rolling_hurst_safe(close_arr, window=100)

# 3. Hurst Amplified Momentum
df['Hurst_Amp_Momentum'] = df['Kinematic_Velocity'] * (df['Hurst'] * 2.0)

# Clean NaNs
df.dropna(subset=['Hurst', 'Hurst_Amp_Momentum'], inplace=True)

# 4. Volume Rolling Mean
df['Vol_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().ffill().fillna(0)

# 5. Newtonian Volume-Momentum Decision Engine
vol_mom_decisions = []
for i in range(len(df)):
    curr_vel = df['Kinematic_Velocity'].iloc[i]
    curr_acc = df['Kinematic_Acceleration'].iloc[i]
    curr_vol = df['Volume'].iloc[i]
    avg_vol = df['Vol_MA_20'].iloc[i]
    
    if curr_vel > 15.0 and curr_acc > 0.0 and curr_vol > avg_vol:
        status = "🟢 NEWTON BULL"
    elif curr_vel < -15.0 and curr_acc < 0.0 and curr_vol > avg_vol:
        status = "🔴 NEWTON BEAR"
    else:
        status = "⚪ NEUTRAL FLAT"
    vol_mom_decisions.append(status)

df['Vol_Mom_Decision'] = vol_mom_decisions

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL
# =====================================================================
df_predict = df.copy()

st.success("🔒 **Leakage Patched!** `np.roll` has been completely deleted. System is now mathematically 100% causal.")

# Clean Layout Display
clean_cols = [
    'Close', 'Kinematic_Velocity', 'Kinematic_Acceleration', 'Volume', 
    'Hurst', 'Hurst_Amp_Momentum', 'Vol_Mom_Decision'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding
display_df['Close_Raw'] = display_df['Close_Raw'].round(2)
display_df['Kinematic_Velocity'] = display_df['Kinematic_Velocity'].round(2)
display_df['Kinematic_Acceleration'] = display_df['Kinematic_Acceleration'].round(4)
display_df['Volume'] = display_df['Volume'].round(0)
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Hour Newtonian Kinematics Analytics Board")
st.dataframe(display_df, use_container_width=True, height=750)
