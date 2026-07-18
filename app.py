import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Page Configuration
st.set_page_config(page_title="Nifty Locked Range Bar Engine", layout="wide")
st.title("⚡ Nifty 50 - 20-Point Range Bar Master Engine")
st.write("🎯 **Pure Price Action:** Frozen Past Data History + Highlighted Active Live Candle")

# =====================================================================
# MATHEMATICAL ENGINES (Exact Copy-Paste Logic)
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
# 🛡️ SYSTEM DATA INGESTION
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live Nifty 50 Data (^NSEI)..."):
    try:
        df = yf.download(tickers="^NSEI", period="2y", interval="1h")
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
            st.error("🚨 Error: Insufficient Nifty data lines.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# 🧱 20-POINT RANGE CANDLE GENERATION ENGINE
# =====================================================================
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index

range_size = 20.0
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

# Create Range Bar DataFrame
df_range = pd.DataFrame(index=range_times)
df_range['Close'] = range_closes

if 'fixed_split_idx' not in st.session_state:
    st.session_state.fixed_split_idx = int(len(df_range) * 0.50)

df_predict = df_range.iloc[st.session_state.fixed_split_idx:].copy()
close_arr = df_predict['Close'].to_numpy(dtype=float)

# =====================================================================
# 📊 DOWNSTREAM SIGNAL CALCULATIONS ON RANGE BARS
# =====================================================================
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

# =====================================================================
# 📊 PURE HAM-BASED PROBABILITY ENGINE
# =====================================================================
rolling_window = 30
mom_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
mom_mean = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).mean().to_numpy()
mom_std = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).std().fillna(1e-6).to_numpy()
mom_std = np.where(mom_std == 0, 1e-6, mom_std)

prob_up, prob_down = [], []
signal_log = []
bar_status = []

for i in range(len(mom_vals)):
    val = mom_vals[i]
    m = mom_mean[i]
    s = mom_std[i]
    
    z_score = (val - m) / s
    p_up = norm.cdf(z_score)
    p_up = np.clip(p_up, 0.01, 0.99)
    
    prob_up.append(round(p_up, 2))
    prob_down.append(round(1.0 - p_up, 2))
    
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

if is_live_candle_running:
    df_predict.iloc[-1, df_predict.columns.get_loc('Signal')] = f"⚡ LIVE ({df_predict['Signal'].iloc[-1].split()[-1]})"

# =====================================================================
# 📋 DASHBOARD METRICS
# =====================================================================
latest_row = df_predict.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Nifty Range Close (INR)", value=f"{latest_row['Close']:.2f}")
with col2:
    st.metric(label="Active HAM Signal", value=f"{latest_row['Signal']}")
with col3:
    st.metric(label="HAM Up Probability", value=f"{latest_row['Prob_Up']*100:.0f}%")
with col4:
    st.metric(label="Bar Lock Status", value=f"{latest_row['Bar_Status']}")

# Matrix Data Display Formulation
clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Signal', 'Prob_Up', 'Prob_Down', 'Bar_Status']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Nifty Close (20-Pt steps)', 'Hurst_Amp_Momentum': 'Raw HAM'}, inplace=True)

display_df['Nifty Close (20-Pt steps)'] = display_df['Nifty Close (20-Pt steps)'].round(2)
display_df['Raw HAM'] = display_df['Raw HAM'].round(4)

# Safely invert and change index format
display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Nifty 20-Point Range Bar Matrix (Locked History vs Active Live)")

# 🚨 SAFETY CRASH CONTROL: Instead of complex Pandas stylers that crash on different python/pandas environments,
# we use standard reliable display with a separate visual banner for the live running state.
if is_live_candle_running:
    st.info(f"🟢 **Active Candle Tracker Running:** Nifty current price is swinging live at **{latest_row['Close']:.2f}** with **{latest_row['Prob_Up']*100:.0f}% Bullish Momentum**.")

st.dataframe(display_df_inverted, use_container_width=True, height=750)
