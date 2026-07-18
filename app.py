import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Page Configuration
st.set_page_config(page_title="BTC Dynamic Range Engine", layout="wide")
st.title("⚡ BTC 200-Point Range Bar Master Engine")
st.write("🎯 **Pure Price Action:** Time-Independent 200-Point Range Candles with Dynamic Adaptive Probability")

# =====================================================================
# MATHEMATICAL ENGINES (Fixed Loop & Real-Time Safe)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: 
        return []
    x, p = data_array[0], initial_p  
    filtered_values = np.zeros(len(data_array))
    for i, z in enumerate(data_array):
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values[i] = x
    return filtered_values

def calculate_rolling_hurst(price_series, window=50):
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
# 🛡️ SYSTEM DATA INGESTION (Bitcoin - 2 Years, 1 Hour Candles)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live Bitcoin Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.ffill().bfill()
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
# 🧱 200-POINT RANGE CANDLE GENERATION ENGINE
# =====================================================================
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index

range_size = 200.0
range_closes = []
range_times = []

current_anchor = raw_closes[0]
range_closes.append(current_anchor)
range_times.append(raw_times[0])

for i in range(1, len(raw_closes)):
    price_diff = raw_closes[i] - current_anchor
    if abs(price_diff) >= range_size:
        num_bars = int(abs(price_diff) // range_size)
        direction = np.sign(price_diff)
        for _ in range(num_bars):
            current_anchor += direction * range_size
            range_closes.append(current_anchor)
            range_times.append(raw_times[i])

is_live_candle_running = raw_closes[-1] != range_closes[-1]
if is_live_candle_running:
    range_closes.append(raw_closes[-1])
    range_times.append(raw_times[-1])

df_range = pd.DataFrame(index=range_times)
df_range['Close'] = range_closes

split_idx = int(len(df_range) * 0.50)
df_predict = df_range.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Processed {len(df_predict)} Adaptive Range Bars!**")
close_arr = df_predict['Close'].to_numpy(dtype=float)

# =====================================================================
# 📊 DOWNSTREAM SIGNAL CALCULATIONS ON RANGE BARS
# =====================================================================
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Hurst_Amp_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1) * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

# =====================================================================
# 🧠 MAGICAL ADAPTIVE PROBABILITY ENGINE (Dynamic Tracking)
# =====================================================================
mom_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
hurst_vals = df_predict['Hurst'].to_numpy()

prob_up, prob_down = [], []
signal_log = []
bar_status = []
active_windows = []

# Minimum aur Maximum limits for memory window
MIN_WIN, MAX_WIN = 20, 80

for i in range(len(mom_vals)):
    # 🌟 Magical Step: Dynamic window selection based on current market state
    h_factor = hurst_vals[i] # ranges between 0 and 1
    # Higher hurst = stronger trend = smaller window needed
    dynamic_w = int(MAX_WIN - (h_factor * (MAX_WIN - MIN_WIN)))
    dynamic_w = max(MIN_WIN, min(MAX_WIN, dynamic_w))
    
    start_idx = max(0, i - dynamic_w + 1)
    window_data = mom_vals[start_idx:i+1]
    
    m = np.mean(window_data)
    s = np.std(window_data) if len(window_data) > 1 else 1e-6
    if s == 0: s = 1e-6
    
    z_score = (mom_vals[i] - m) / s
    p_up = norm.cdf(z_score)
    p_up = np.clip(p_up, 0.01, 0.99)
    
    prob_up.append(round(p_up, 2))
    prob_down.append(round(1.0 - p_up, 2))
    active_windows.append(dynamic_w)
    
    if p_up > 0.50:
        signal_log.append("🟢 BUY")
    else:
        signal_log.append("🔴 SELL")
        
    if i == len(mom_vals) - 1 and is_live_candle_running:
        bar_status.append("🔄 LIVE ACTIVE")
    else:
        bar_status.append("🔒 FROZEN")

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down
df_predict['Signal'] = signal_log
df_predict['Bar_Status'] = bar_status
df_predict['Dynamic_Window'] = active_windows

if is_live_candle_running and len(df_predict) > 0:
    current_sig = df_predict['Signal'].iloc[-1].split()[-1]
    df_predict.iloc[-1, df_predict.columns.get_loc('Signal')] = f"⚡ LIVE ({current_sig})"

# =====================================================================
# 📋 DASHBOARD METRICS & TABLE
# =====================================================================
latest_row = df_predict.iloc[-1]
delta_close = f"${(latest_row['Close'] - df_predict['Close'].iloc[-2]):.2f}" if len(df_predict) > 1 else "$0.00"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="BTC Range Close (USD)", value=f"${latest_row['Close']:.2f}", delta=delta_close)
with col2:
    st.metric(label="Active HAM Signal", value=f"{latest_row['Signal']}")
with col3:
    st.metric(label="Adaptive Probability", value=f"{latest_row['Prob_Up']*100:.0f}%", delta=f"Based on {latest_row['Dynamic_Window']} Bars")
with col4:
    st.metric(label="Current Engine Memory", value=f"{latest_row['Dynamic_Window']} Bars", help="Changes shape automatically using Hurst Index")

clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Signal', 'Prob_Up', 'Prob_Down', 'Dynamic_Window', 'Bar_Status']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'BTC Close', 'Hurst_Amp_Momentum': 'Raw HAM', 'Dynamic_Window': 'Window Used'}, inplace=True)

display_df['BTC Close'] = display_df['BTC Close'].round(2)
display_df['Raw HAM'] = display_df['Raw HAM'].round(4)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 200-Point Adaptive Matrix (Self-Adjusting Engine)")
st.dataframe(display_df_inverted, use_container_width=True, height=750)
