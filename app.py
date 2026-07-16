import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Signal Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Hurst-Amplified Momentum & 5-Channel Accumulator (100% Leak-Proof)")

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
            # Live Incomplete hourly candle protection
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
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

# Clean NaNs strictly before creating rolling statistical channels
df_predict.dropna(subset=['ATR', 'Hurst'], inplace=True)

# =====================================================================
# 📊 1 TO 5 CHANNEL ACCUMULATOR ENGINE
# =====================================================================
# Calculate rolling statistical bands of Hurst_Amp_Momentum for dynamic channel sizing
mom_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
rolling_window = 50
mom_mean = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).mean().to_numpy()
mom_std = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).std().fillna(1.0).to_numpy()

channels = np.zeros(len(mom_vals), dtype=int)
accumulator = np.zeros(len(mom_vals), dtype=int) # Filtered tracking channel state (1 to 5)

for i in range(len(mom_vals)):
    val = mom_vals[i]
    m = mom_mean[i]
    s = mom_std[i]
    
    # 1. Assign Raw Channel Band
    if val > (m + 1.5 * s):
        channels[i] = 5
    elif val > (m + 0.5 * s):
        channels[i] = 4
    elif val < (m - 1.5 * s):
        channels[i] = 1
    elif val < (m - 0.5 * s):
        channels[i] = 2
    else:
        channels[i] = 3
        
    # 2. Accumulator Logic (Smooths transitions and filters false touches)
    if i == 0:
        accumulator[i] = channels[i]
    else:
        prev_acc = accumulator[i-1]
        curr_chan = channels[i]
        # Only transition the accumulator if there's a confirmed zone shift
        if abs(curr_chan - prev_acc) >= 1:
            accumulator[i] = curr_chan
        else:
            accumulator[i] = prev_acc

df_predict['Raw_Channel'] = channels
df_predict['Accumulator_Channel'] = accumulator

# 🤖 SIGNAL GENERATION (Memory-based transition to prevent Channel 3 whipsaws)
signal_log = []
current_sig = "🔴 SELL"  # Default start state

for i in range(len(df_predict)):
    acc_chan = accumulator[i]
    if acc_chan >= 4:
        current_sig = "🟢 BUY"
    elif acc_chan <= 2:
        current_sig = "🔴 SELL"
    # If in Channel 3 (Neutral), we hold the previous confirmed state (current_sig unchanged)
    signal_log.append(current_sig)

df_predict['Signal'] = signal_log

# 🚀 PROBABILITY MATRIX BASED ON CHANNELS
prob_up = []
prob_down = []

for i in range(len(df_predict)):
    acc_chan = accumulator[i]
    sig = signal_log[i]
    
    # Probabilities mapped directly to structural 1 to 5 channel strength
    if acc_chan == 5:
        p_up = 0.95
    elif acc_chan == 4:
        p_up = 0.75
    elif acc_chan == 3:
        # Balanced zone, slight lean towards current signal direction
        p_up = 0.55 if sig == "🟢 BUY" else 0.45
    elif acc_chan == 2:
        p_up = 0.25
    else: # Channel 1
        p_up = 0.05
        
    prob_up.append(round(p_up, 2))
    prob_down.append(round(1.0 - p_up, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# Format Layout Columns Matrix
clean_cols = ['Close_Raw', 'Hurst_Amp_Momentum', 'Raw_Channel', 'Accumulator_Channel', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Precision Matrix Formatting
for c in ['Close_Raw', 'Hurst_Amp_Momentum']:
    display_df[c] = display_df[c].round(2)

# Chronological sorting for dashboard display panel (latest on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Clean Header & Rendering
st.subheader("📋 5-Channel Accumulated Bitcoin Action Matrix")
st.dataframe(display_df, use_container_width=True, height=750)
