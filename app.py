import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Signal Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** 100% Leakage-Free Magic Trend Intensity Engine")

# =====================================================================
# MATHEMATICAL ENGINES (Fixed Loop & Real-Time Safe)
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
# 🛡️ SYSTEM DATA INGESTION
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live BTC Data..."):
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
            st.error("🚨 Error: Insufficient data lines.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# 1. Split Data FIRST to avoid lookahead bias across the whole dataset
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Candles (No Leakage)!**")

# 2. Apply math ONLY on the isolated prediction slice
df_predict['Close_Raw'] = df_predict['Close']
close_arr = df_predict['Close_Raw'].values

# Strict Past-to-Present Kalman Filter
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# ATR without backward filling (.bfill removed to kill future leakage)
high_low = df_predict['High'] - df_predict['Low']
high_close = np.abs(df_predict['High'] - df_predict['Close'].shift(1))
low_close = np.abs(df_predict['Low'] - df_predict['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_predict['ATR'] = true_range.rolling(14).mean().ffill() 

# Hurst Memory Calculation
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Generate Standard Weighted Momentum Matrix
raw_weighted_momentum = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# Clean Volatility-Normalized ATR-Weighted Momentum
df_predict['ATR_Weighted_Momentum'] = df_predict['Weighted_Momentum'] / (df_predict['ATR'] + 1e-8)

# Magic Indicator (Purely Real-Time)
df_predict['Magic_Trend_Intensity'] = df_predict['ATR_Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

# Drop any initial NaN rows created by rolling windows securely
df_predict.dropna(subset=['ATR', 'Hurst'], inplace=True)

# 🤖 PURE SIGNAL ENGINE LOGIC
magic_arr = df_predict['Magic_Trend_Intensity'].to_numpy()
hurst_arr = df_predict['Hurst'].to_numpy()

signal_log = []
for idx in range(len(df_predict)):
    if magic_arr[idx] >= 0:
        signal_log.append("🟢 BUY")
    else:
        signal_log.append("🔴 SELL")

df_predict['Signal'] = signal_log

# 🚀 HIGH CONTRAST PROBABILITIES
prob_up = []
prob_down = []
for idx in range(len(df_predict)):
    sig = signal_log[idx]
    magic_val = magic_arr[idx]
    
    intensity = min(0.49, abs(magic_val) * 0.15) 
    
    if sig == "🟢 BUY":
        p_up = round(0.50 + intensity, 2)
        if p_up < 0.55: p_up = 0.58
        prob_up.append(p_up)
        prob_down.append(round(1.0 - p_up, 2))
    else:
        p_down = round(0.50 + intensity, 2)
        if p_down < 0.55: p_down = 0.58 
        prob_down.append(p_down)
        prob_up.append(round(1.0 - p_down, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix
clean_cols = ['Close_Raw', 'Kalman_Baseline', 'Hurst', 'ATR', 'Weighted_Momentum', 'ATR_Weighted_Momentum', 'Magic_Trend_Intensity', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Roundings
for c in ['Close_Raw', 'Kalman_Baseline', 'ATR', 'Weighted_Momentum', 'ATR_Weighted_Momentum', 'Magic_Trend_Intensity']:
    display_df[c] = display_df[c].round(2)
for c in ['Hurst', 'Prob_Up', 'Prob_Down']:
    display_df[c] = display_df[c].round(3)

# Reverse for latest candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Live Actionable Bitcoin Master Matrix (Anti-Leakage Mode)")
st.dataframe(display_df, use_container_width=True, height=750)
