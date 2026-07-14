import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Volatility Regime Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Solid Regime Engine")
st.write("🎯 **Anti-Chop Framework:** Hurst Exponent + ATR Filter | Zero Noise Signals")

# =====================================================================
# MATHEMATICAL ENGINES (Kalman, Hurst, and ATR)
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
    """Calculates rolling Hurst Exponent to identify Choppy vs Trending regimes"""
    hurst_values = np.full(len(price_series), 0.5) # Default to random walk
    log_returns = np.log(price_series / np.roll(price_series, 1))
    log_returns[0] = 0
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i-window+1:i+1]
        # Calculate accumulated deviations
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        rs_ratio = r_val / s_val
        
        # Approximate Hurst Exponent via linear fit log(R/S) / log(n)
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

# Calculate Noise-Filtered Baseline
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# Calculate ATR (Average True Range)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill().bfill()
df['Volatility_Ratio'] = df['ATR'] / df['Close_Raw']

# Calculate Market Memory (Hurst Exponent)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Target Engine Setup
df['Target_Next_Direction'] = np.where(df['Close_Raw'].shift(-1) > df['Close_Raw'], 1, 0)

# Split Engine 50:50
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

# 🤖 ADVANCED REGIME SIGNALS (Anti-Chop Core)
hurst_arr = df_predict['Hurst'].to_numpy()
vol_ratio = df_predict['Volatility_Ratio'].to_numpy()
raw_close = df_predict['Close_Raw'].to_numpy()
kalman_line = df_predict['Kalman_Baseline'].to_numpy()

signal_log = []
for idx in range(len(df_predict)):
    # 1. Condition for Choppy Market (Hurst between 0.42 and 0.53 means random walk/consolidation)
    if 0.42 <= hurst_arr[idx] <= 0.54:
        signal_log.append("⚠️ CHOPPY ZONE")
    else:
        # 2. Trending Market Logic
        if raw_close[idx] > kalman_line[idx] and hurst_arr[idx] > 0.54:
            signal_log.append("🟢 BUY")
        elif raw_close[idx] < kalman_line[idx] and hurst_arr[idx] > 0.54:
            signal_log.append("🔴 SELL")
        # 3. Mean Reversion Logic (Low Hurst means price will bounce back to baseline)
        elif hurst_arr[idx] < 0.42:
            if raw_close[idx] < (kalman_line[idx] - df_predict['ATR'].iloc[idx]):
                signal_log.append("🟢 BUY (REVERSAL)")
            elif raw_close[idx] > (kalman_line[idx] + df_predict['ATR'].iloc[idx]):
                signal_log.append("🔴 SELL (REVERSAL)")
            else:
                signal_log.append("⚠️ CHOPPY ZONE")
        else:
            signal_log.append("⚠️ CHOPPY ZONE")

df_predict['Signal'] = signal_log

# 🚀 HIGH CONTRAST PROBABILITIES SIMULATION BASED ON REGIME CONFIDENCE
prob_up = []
prob_down = []
for idx in range(len(df_predict)):
    sig = signal_log[idx]
    if "BUY" in sig:
        prob_up.append(0.99 if hurst_arr[idx] > 0.58 else 0.85)
        prob_down.append(0.01 if hurst_arr[idx] > 0.58 else 0.15)
    elif "SELL" in sig:
        prob_up.append(0.01 if hurst_arr[idx] > 0.58 else 0.15)
        prob_down.append(0.99 if hurst_arr[idx] > 0.58 else 0.85)
    else: # Choppy Zone
        prob_up.append(0.50)
        prob_down.append(0.50)

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix
clean_cols = ['Close_Raw', 'Kalman_Baseline', 'Hurst', 'ATR', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Roundings
for c in ['Close_Raw', 'Kalman_Baseline', 'ATR']:
    display_df[c] = display_df[c].round(2)
for c in ['Hurst', 'Prob_Up', 'Prob_Down']:
    display_df[c] = display_df[c].round(3)

# Reverse for latest candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader(f"📋 Live Anti-Chop Bitcoin Master Matrix")
st.dataframe(display_df, use_container_width=True, height=750)
