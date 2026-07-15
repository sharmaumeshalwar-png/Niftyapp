import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Master Signal Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Volatility-Normalized ATR Momentum Engine + Dynamic Confidence Matrix")

# =====================================================================
# MATHEMATICAL ENGINES (Kalman, Hurst, ATR, and Momentum Matrix)
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
# 🛡️ SYSTEM DATA INGESTION (2-YEAR HIGH DENSITY STREAM)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching 2-Year Hourly Live BTC Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient lines from server.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

st.success(f"🟢 **Synced {len(df)} Real-Time Bitcoin 1-Hour Candles (IST)!**")

# Setup Primary Math
df['Close_Raw'] = df['Close']
close_arr = df['Close_Raw'].values

df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# Volatility Band Extraction (ATR)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill().bfill()

# ATR Kalman Filter (initial_p = 0.50)
df['ATR_Kalman'] = apply_kalman_filter_custom(df['ATR'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

# Hurst Memory Calculation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Generate Standard Weighted Momentum Matrix
raw_weighted_momentum = df['Close_Raw'] - df['Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 🆕 PERFECTING THE INDICATOR: Volatility-Normalized ATR-Weighted Momentum
# E.g., Momentum ko Volatility (ATR) se balance kiya taaki real price drive measure ho sake
df['ATR_Weighted_Momentum'] = df['Weighted_Momentum'] / (df['ATR_Kalman'] + 1e-8)

# Split Engine 50:50
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

# 🤖 PURE SIGNAL ENGINE LOGIC (Based on new ATR Weighted Momentum)
hurst_arr = df_predict['Hurst'].to_numpy()
atr_wm_arr = df_predict['ATR_Weighted_Momentum'].to_numpy()

signal_log = []
for idx in range(len(df_predict)):
    # Macro Trend + Local Resolution unified via ATR Weighted Momentum
    if atr_wm_arr[idx] >= 0:
        signal_log.append("🟢 BUY")
    else:
        signal_log.append("🔴 SELL")

df_predict['Signal'] = signal_log

# 🚀 🆕 HIGH CONTRAST PROBABILITIES GENERATION BASED ON ATR WEIGHTED MOMENTUM
prob_up = []
prob_down = []
for idx in range(len(df_predict)):
    sig = signal_log[idx]
    mom_val = atr_wm_arr[idx]
    
    # Adaptive Dynamic Scaling: Momentum jitna strong, confidence utna high (Cap at 0.99)
    intensity = min(0.49, abs(mom_val) * 0.1) 
    
    if sig == "🟢 BUY":
        p_up = round(0.50 + intensity, 2)
        # Extra boost if Hurst confirms trending structure
        if hurst_arr[idx] > 0.52 and p_up < 0.90:
            p_up = min(0.99, p_up + 0.10)
        prob_up.append(p_up)
        prob_down.append(round(1.0 - p_up, 2))
    else:
        p_down = round(0.50 + intensity, 2)
        if hurst_arr[idx] > 0.52 and p_down < 0.90:
            p_down = min(0.99, p_down + 0.10)
        prob_down.append(p_down)
        prob_up.append(round(1.0 - p_down, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix (Including ATR, ATR_Kalman and New ATR Weighted Momentum)
clean_cols = ['Close_Raw', 'Kalman_Baseline', 'Hurst', 'ATR', 'ATR_Kalman', 'Weighted_Momentum', 'ATR_Weighted_Momentum', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Roundings for clean dashboard display
for c in ['Close_Raw', 'Kalman_Baseline', 'ATR', 'ATR_Kalman', 'Weighted_Momentum', 'ATR_Weighted_Momentum']:
    display_df[c] = display_df[c].round(2)
for c in ['Hurst', 'Prob_Up', 'Prob_Down']:
    display_df[c] = display_df[c].round(3)

# Reverse for latest candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Live Actionable Bitcoin Master Matrix")
st.dataframe(display_df, use_container_width=True, height=750)
