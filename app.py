import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Signal Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Hurst-Amplified Momentum & Trailing Accumulator (100% Leak-Proof Magical Mode)")

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

# 🔥 CRITICAL DATA SCIENCE RULE: Split data FIRST to prevent lookahead leakage globally
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Candles (No Leakage)!**")

# Setup Isolated Price Arrays
df_predict['Close_Raw'] = df_predict['Close']
close_arr = df_predict['Close_Raw'].values

# Strict Isolated Price Kalman Baseline Calculation
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# ATR calculation without lookahead/bfill bias
high_low = df_predict['High'] - df_predict['Low']
high_close = np.abs(df_predict['High'] - df_predict['Close'].shift(1))
low_close = np.abs(df_predict['Low'] - df_predict['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_predict['ATR'] = true_range.rolling(14).mean().ffill() 

# Hurst Vector Generation on isolated window
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Exact original Price-based Weighted Momentum Calculation
raw_weighted_momentum = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 🔥 THE MAGICAL MULTIPLICATION: Scaling Momentum by Hurst Intensity
# Multiplying Hurst by 2 to normalize around 1.0 when market is balanced at 0.50
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

# Drop initial setup NaNs safely (strictly no bfill cheat allowed)
df_predict.dropna(subset=['ATR', 'Hurst'], inplace=True)

# =====================================================================
# ⚙️ EXPONENTIAL ACCUMULATOR & DYNAMIC PIVOT ENGINE ON AMP MOMENTUM
# =====================================================================
amp_mom_arr = df_predict['Hurst_Amp_Momentum'].to_numpy()
alpha = 0.15  # Accumulation factor

accumulator = np.zeros(len(amp_mom_arr))
pivot_state = np.zeros(len(amp_mom_arr))  # 1 for bullish momentum rise, -1 for bearish fall

# Initial setting
accumulator[0] = amp_mom_arr[0]
pivot_state[0] = 1 if amp_mom_arr[0] >= 0 else -1

peak_val = accumulator[0]
valley_val = accumulator[0]

for idx in range(1, len(amp_mom_arr)):
    # 1. Exponential Accumulation of the Magical Hurst-Amplified Momentum
    accumulator[idx] = (alpha * amp_mom_arr[idx]) + ((1.0 - alpha) * accumulator[idx-1])
    
    # Track localized peak and valley points for trailing pivots
    if pivot_state[idx-1] == 1:
        if accumulator[idx] > peak_val:
            peak_val = accumulator[idx]
        # 2. Dynamic Pivot Check (Paltate hi ghumna chahiye)
        pivot_threshold = 0.20 * abs(peak_val) if abs(peak_val) > 1.0 else 0.5
        if accumulator[idx] < (peak_val - pivot_threshold):
            pivot_state[idx] = -1  # Turn Bearish
            valley_val = accumulator[idx]
        else:
            pivot_state[idx] = 1   # Continue Riding Up
    else:
        if accumulator[idx] < valley_val:
            valley_val = accumulator[idx]
        # Pivot check for upside turn
        pivot_threshold = 0.20 * abs(valley_val) if abs(valley_val) > 1.0 else 0.5
        if accumulator[idx] > (valley_val + pivot_threshold):
            pivot_state[idx] = 1   # Turn Bullish
            peak_val = accumulator[idx]
        else:
            pivot_state[idx] = -1  # Continue Riding Down

df_predict['Accumulator'] = accumulator
df_predict['Pivot_State'] = pivot_state

# 🤖 PURE SIGNAL ENGINE LOGIC (Triggered directly by the magical engine state)
signal_log = []
for idx in range(len(df_predict)):
    if pivot_state[idx] == 1:
        signal_log.append("🟢 BUY")
    else:
        signal_log.append("🔴 SELL")

df_predict['Signal'] = signal_log

# 🚀 HIGH CONTRAST PROBABILITIES GENERATION
prob_up = []
prob_down = []
for idx in range(len(df_predict)):
    sig = signal_log[idx]
    acc_val = abs(accumulator[idx])
    
    # Probabilities scale organically based on Hurst-Amplified Accumulator expansion
    prob_mod = min(0.42, acc_val * 0.03)
    base_probability = 0.55
    
    if sig == "🟢 BUY":
        p_up = round(base_probability + prob_mod, 2)
        p_up = min(0.99, p_up)
        prob_up.append(p_up)
        prob_down.append(round(1.0 - p_up, 2))
    else:
        p_down = round(base_probability + prob_mod, 2)
        p_down = min(0.99, p_down)
        prob_down.append(p_down)
        prob_up.append(round(1.0 - p_down, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix
clean_cols = ['Close_Raw', 'Kalman_Baseline', 'Hurst', 'ATR', 'Weighted_Momentum', 'Hurst_Amp_Momentum', 'Accumulator', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Precision Matrix Formatting
for c in ['Close_Raw', 'Kalman_Baseline', 'ATR', 'Weighted_Momentum', 'Hurst_Amp_Momentum', 'Accumulator']:
    display_df[c] = display_df[c].round(2)
for c in ['Hurst', 'Prob_Up', 'Prob_Down']:
    display_df[c] = display_df[c].round(3)

# Chronological sorting for dashboard display panel (latest on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Clean Header & Rendering
st.subheader("📋 Live Actionable Bitcoin Master Matrix (Hurst-Amplified Magical Engine)")
st.dataframe(display_df, use_container_width=True, height=750)
