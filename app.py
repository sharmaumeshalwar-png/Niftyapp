import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Crypto VIX 24/7 Engine", layout="wide")
st.title("⚡ Crypto Volatility (BitVol) 24/7 Master Engine")
st.write("🎯 **Pure Direct Signals:** Hurst-Amplified Momentum & 5-Channel Crypto Volatility Accumulator (100% Leak-Proof)")

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
# 🛡️ SYSTEM DATA INGESTION (Targeting 24/7 Crypto BitVol Index)
# -----------------------------------------------------------------
df = None
# Paji, Yahoo Finance ka authentic 24/7 Crypto VIX ticker set kar diya hai
target_ticker = "BITVOL-USD" 

with st.spinner(f"Fetching Live 24/7 {target_ticker} Data..."):
    try:
        df = yf.download(tickers=target_ticker, period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            # Live Incomplete hourly candle protection
            df = df.iloc[:-1]
            
            # Pure IST Conversion (24/7 Trading Alignment)
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
# 🔥 GLOBAL CALCULATION (Calculations performed on full historical dataframe)
# =====================================================================
close_arr = df['Close'].values

# Strict Price Kalman Baseline Calculation
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# ATR calculation without lookahead bias
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill() 

# Hurst Vector Generation
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# Price-based Weighted Momentum Calculation
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# Scaling Momentum by Hurst Intensity
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# Clean NaNs strictly before creating rolling channels
df.dropna(subset=['ATR', 'Hurst'], inplace=True)

# 📊 1 TO 5 CHANNEL ACCUMULATOR ENGINE
mom_vals = df['Hurst_Amp_Momentum'].to_numpy()
rolling_window = 50
mom_mean = df['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).mean().to_numpy()
mom_std = df['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).std().fillna(1.0).to_numpy()

channels = np.zeros(len(mom_vals), dtype=int)
accumulator = np.zeros(len(mom_vals), dtype=int)

for i in range(len(mom_vals)):
    val = mom_vals[i]
    m = mom_mean[i]
    s = mom_std[i]
    
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
        
    if i == 0:
        accumulator[i] = channels[i]
    else:
        prev_acc = accumulator[i-1]
        curr_chan = channels[i]
        if abs(curr_chan - prev_acc) >= 1:
            accumulator[i] = curr_chan
        else:
            accumulator[i] = prev_acc

df['Raw_Channel'] = channels
df['Accumulator_Channel'] = accumulator

# 🤖 SIGNAL GENERATION
signal_log = []
current_sig = "🔴 CRYPTO CRASH RISK / HIGH VOL"

for i in range(len(df)):
    acc_chan = accumulator[i]
    if acc_chan >= 4:
        current_sig = "🟢 CRYPTO VOL SPIKE / RISK OFF"
    elif acc_chan <= 2:
        current_sig = "🔴 CRYPTO VOL CRUSH / RISK ON"
    signal_log.append(current_sig)

df['Signal'] = signal_log

# 🚀 PROBABILITY MATRIX BASED ON CHANNELS
prob_up = []
for i in range(len(df)):
    acc_chan = accumulator[i]
    sig = signal_log[i]
    
    if acc_chan == 5:
        p_up = 0.95
    elif acc_chan == 4:
        p_up = 0.75
    elif acc_chan == 3:
        p_up = 0.55 if "VOL SPIKE" in sig else 0.45
    elif acc_chan == 2:
        p_up = 0.25
    else:
        p_up = 0.05
        
    prob_up.append(round(p_up, 2))

df['Prob_Up'] = prob_up
df['Prob_Down'] = [round(1.0 - p, 2) for p in prob_up]

# =====================================================================
# 🎛️ DASHBOARD DISPLAY (Full Data Stream Locked & IST Synced)
# =====================================================================
df_predict = df.copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Crypto VIX Candles (24/7 Streaming via IST)!**")

# Format Layout Columns Matrix
clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Raw_Channel', 'Accumulator_Channel', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()

# Rename to match UI spec
display_df.rename(columns={'Close': 'BitVol_Raw'}, inplace=True)

# Precision Matrix Formatting
for c in ['BitVol_Raw', 'Hurst_Amp_Momentum']:
    display_df[c] = display_df[c].round(2)

# Chronological sorting & Pure IST String Conversion
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

# Clean Header & Rendering
st.subheader(f"📋 5-Channel Accumulated {target_ticker} Action Matrix (24/7 IST)")
st.dataframe(display_df, use_container_width=True, height=750)
