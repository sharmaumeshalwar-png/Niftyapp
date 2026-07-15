import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Signal Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Kalman Filtered ATR Weighted Momentum (100% Leak-Proof)")

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

# 🔥 CRITICAL FIX 1: Split data FIRST to prevent Lookahead/Information Leakage
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Candles (No Leakage)!**")

# Setup Isolated Price Arrays
df_predict['Close_Raw'] = df_predict['Close']
close_arr = df_predict['Close_Raw'].values

# Strict Isolated Price Kalman Baseline
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# Raw ATR Calculation (strictly no .bfill lookahead allowed)
high_low = df_predict['High'] - df_predict['Low']
high_close = np.abs(df_predict['High'] - df_predict['Close'].shift(1))
low_close = np.abs(df_predict['Low'] - df_predict['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_predict['ATR'] = true_range.rolling(14).mean().ffill() 

# Hurst Vector Generation on isolated window
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Drop initial NaNs securely before running Kalman on ATR (to keep array clean)
df_predict.dropna(subset=['ATR', 'Hurst'], inplace=True)

# 🔥 CRITICAL FIX 2: Apply Kalman Filter strictly on the ATR values (not price)
atr_arr = df_predict['ATR'].values
df_predict['ATR_Kalman'] = apply_kalman_filter_custom(atr_arr, initial_p=10.0, q_val=0.001, r_val=0.1)

# 🔥 CRITICAL FIX 3: Weighted Momentum is now ATR minus smoothed ATR_Kalman
df_predict['Weighted_Momentum'] = df_predict['ATR'] - df_predict['ATR_Kalman']

# 🤖 PURE SIGNAL ENGINE LOGIC (Based on price trend & ATR Weighted Momentum status)
hurst_arr = df_predict['Hurst'].to_numpy()
raw_close = df_predict['Close_Raw'].to_numpy()
kalman_line = df_predict['Kalman_Baseline'].to_numpy()
atr_momentum = df_predict['Weighted_Momentum'].to_numpy()

signal_log = []
for idx in range(len(df_predict)):
    # 1. Macro Trend Regime (Hurst > 0.52): Trend following price alignment with ATR momentum confirmation
    if hurst_arr[idx] > 0.52:
        if raw_close[idx] > kalman_line[idx] and atr_momentum[idx] >= 0:
            signal_log.append("🟢 BUY")
        else:
            signal_log.append("🔴 SELL")
    # 2. Sideways Regime (Hurst <= 0.52): Local dynamic mean reversion
    else:
        deviation = raw_close[idx] - kalman_line[idx]
        if deviation >= 0:
            signal_log.append("🟢 BUY")
        else:
            signal_log.append("🔴 SELL")

df_predict['Signal'] = signal_log

# 🚀 HIGH CONTRAST PROBABILITIES GENERATION BASED ON ATR MOMENTUM
prob_up = []
prob_down = []
for idx in range(len(df_predict)):
    sig = signal_log[idx]
    h_factor = hurst_arr[idx]
    mom_val = abs(atr_momentum[idx])
    
    # Confidence scaling based on the intensity of ATR baseline momentum
    confidence_shift = min(0.40, mom_val * 0.05)
    base_probability = 0.58 if h_factor > 0.55 else 0.52
    
    if sig == "🟢 BUY":
        p_up = round(base_probability + confidence_shift, 2)
        p_up = min(0.99, p_up)
        prob_up.append(p_up)
        prob_down.append(round(1.0 - p_up, 2))
    else:
        p_down = round(base_probability + confidence_shift, 2)
        p_down = min(0.99, p_down)
        prob_down.append(p_down)
        prob_up.append(round(1.0 - p_down, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix
clean_cols = ['Close_Raw', 'Kalman_Baseline', 'Hurst', 'ATR', 'ATR_Kalman', 'Weighted_Momentum', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Precision Matrix Formatting
for c in ['Close_Raw', 'Kalman_Baseline', 'ATR', 'ATR_Kalman', 'Weighted_Momentum']:
    display_df[c] = display_df[c].round(2)
for c in ['Hurst', 'Prob_Up', 'Prob_Down']:
    display_df[c] = display_df[c].round(3)

# Chronological sorting for display panel (latest on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Clean Header & Rendering
st.subheader("📋 Live Actionable Bitcoin Master Matrix (ATR-Based Momentum Engine)")
st.dataframe(display_df, use_container_width=True, height=750)
