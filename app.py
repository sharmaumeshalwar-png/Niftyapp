import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Strict Zero-Leakage Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Strict Divergence Engine")
st.write("🎯 **Pure Value Trading:** 1-Hour Candles | Kalman (0.850) & 1-Candle WMA (Raw) | 100% Zero-Leakage Locked")

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

def apply_kalman_adaptive_p(data_array, p_array, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: return []
    x = data_array[0]
    filtered_values = []
    for z, p_init in zip(data_array, p_array):
        p = p_init + q_val
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
# 🛡️ STRICT DATA INGESTION & RUNNING CANDLE PURGE
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live 1-Year BTC Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="1y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
            # ⛔ CUTOFF ENGINE: Immediately drop the active, unclosed 1-hour candle.
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

# Price Kalman Baseline
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# ATR calculation (Using historical shift only)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill().fillna(0)

# Hurst Vector Generation (Strictly historical window)
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Price-based Weighted Momentum
df['Close_Minus_Kalman'] = df['Close'] - df['Kalman_Baseline']
raw_weighted_momentum = df['Close_Minus_Kalman'].values
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum, initial_p=0.50, q_val=0.001, r_val=0.1)

# Hurst Amplification
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# Clean dynamic NaNs before sequential math operations
df.dropna(subset=['ATR', 'Hurst'], inplace=True)

# =====================================================================
# 🧮 CUSTOM MATH COLUMNS, KALMANS & WMA OPERATIONS
# =====================================================================
# 1. Base ATR * Hurst Column
df['ATR_x_Hurst'] = df['ATR'] * df['Hurst']

# 2. Kalman Filter on ATR * Hurst (Strictly initial_p = 0.850 as requested)
atr_hurst_vals = df['ATR_x_Hurst'].ffill().fillna(0).values
df['Kalman_ATR_Hurst'] = apply_kalman_filter_custom(atr_hurst_vals, initial_p=0.850, q_val=0.001, r_val=0.1)

# 3. 1-Candle WMA on ATR * Hurst (1-candle window is mathematically equal to the raw value itself)
df['WMA_ATR_Hurst'] = df['ATR_x_Hurst'].ffill().fillna(0)

# Original Custom Columns
df['Column_A'] = df['Hurst'] * (df['High'] - df['Low'])
df['Column_B'] = df['Column_A'] * df['Hurst_Amp_Momentum']

# Safe conversion with Forward Fill (ffill) and fillna(0) to block any lookahead gaps
col_a_vals = df['Column_A'].ffill().fillna(0).values
col_b_vals = df['Column_B'].ffill().fillna(0).values
atr_vals = df['ATR'].ffill().fillna(0).values

# Adaptive Kalmans on Column A and Column B
df['Kalman_Column_A'] = apply_kalman_adaptive_p(col_a_vals, atr_vals, q_val=0.001, r_val=0.1)
df['Kalman_Column_B'] = apply_kalman_adaptive_p(col_b_vals, atr_vals, q_val=0.001, r_val=0.1)

# =====================================================================
# 🛡️ STRICT 2-STATE DIVERGENCE DETECTION (BTX vs TRAP)
# =====================================================================
df['Std_KA'] = df['Kalman_Column_A'].rolling(20, min_periods=1).std().fillna(1.0)
df['Std_KB'] = df['Kalman_Column_B'].rolling(20, min_periods=1).std().fillna(1.0)

btc_status_list = ["Initializing"]

for i in range(1, len(df)):
    curr_ka = df['Kalman_Column_A'].iloc[i]
    prev_ka = df['Kalman_Column_A'].iloc[i-1]
    curr_kb = df['Kalman_Column_B'].iloc[i]
    prev_kb = df['Kalman_Column_B'].iloc[i-1]
    
    thresh_a = 0.15 * df['Std_KA'].iloc[i]
    thresh_b = 0.15 * df['Std_KB'].iloc[i]
    
    diff_a = curr_ka - prev_ka
    diff_b = curr_kb - prev_kb
    
    dir_a = 0
    if abs(diff_a) > thresh_a:
        dir_a = 1 if diff_a > 0 else -1
        
    dir_b = 0
    if abs(diff_b) > thresh_b:
        dir_b = 1 if diff_b > 0 else -1
        
    if dir_a != 0 and dir_b != 0 and dir_a != dir_b:
        status = "🟩 BTX"
    else:
        status = "⚠️ TRAP"
        
    btc_status_list.append(status)

df['BTC_Status'] = btc_status_list

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Zero Repaint Frozen Layout)
# =====================================================================
df_predict = df.copy()

st.success("🟢 **Pipeline Locked:** Kalman Filter calibrated to initial_p=0.850. WMA limited to 1-Candle raw resolution.")

# Display grid
clean_cols = [
    'Close', 'High', 'Low', 'ATR', 'Hurst', 
    'ATR_x_Hurst', 'Kalman_ATR_Hurst', 'WMA_ATR_Hurst', 'Hurst_Amp_Momentum', 
    'Column_A', 'Kalman_Column_A', 'Column_B', 'Kalman_Column_B', 'BTC_Status'
]
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Close_Raw', 'Hurst': 'Hurst_Value'}, inplace=True)

# Precision Rounding
display_df['Hurst_Value'] = display_df['Hurst_Value'].round(4)
display_df['ATR_x_Hurst'] = display_df['ATR_x_Hurst'].round(4)
display_df['Kalman_ATR_Hurst'] = display_df['Kalman_ATR_Hurst'].round(4)
display_df['WMA_ATR_Hurst'] = display_df['WMA_ATR_Hurst'].round(4)
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)
display_df['Column_A'] = display_df['Column_A'].round(4)
display_df['Kalman_Column_A'] = display_df['Kalman_Column_A'].round(4)
display_df['Column_B'] = display_df['Column_B'].round(4)
display_df['Kalman_Column_B'] = display_df['Kalman_Column_B'].round(4)
for c in ['Close_Raw', 'High', 'Low', 'ATR']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Bitcoin 1-Year Strict Real-Time Trading Matrix")
st.dataframe(display_df, use_container_width=True, height=750)
