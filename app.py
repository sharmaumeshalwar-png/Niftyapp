import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Page Configuration
st.set_page_config(page_title="BTC Kalman Crossover Engine", layout="wide")
st.title("⚡ BTC 200-Point Range Bar Master Engine")
st.write("🎯 **Pure Price Action:** 200-Point Range Candles with Dual Kalman Crossover on Original HAM")

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

st.success(f"🟢 **Synced & Processed {len(df_predict)} Range Bars!**")
close_arr = df_predict['Close'].to_numpy(dtype=float)

# =====================================================================
# 📊 ORIGINAL DOWNSTREAM SIGNAL CALCULATIONS (100% RESTORED 🎯)
# =====================================================================
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)

# Here is your original untouched HAM Value formula
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)

# =====================================================================
# 🎯 DUAL KALMAN CROSSOVER APPLIED DIRECTLY ON ORIGINAL HAM
# =====================================================================
ham_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()

# 1. Fast Kalman applied on your original HAM array
df_predict['Kalman_HAM_Fast'] = apply_kalman_filter_custom(ham_vals, initial_p=1.0, q_val=0.01, r_val=0.05)

# 2. Slow Kalman applied on your original HAM array
df_predict['Kalman_HAM_Slow'] = apply_kalman_filter_custom(ham_vals, initial_p=1.0, q_val=0.0005, r_val=0.5)

fast_kf = df_predict['Kalman_HAM_Fast'].to_numpy()
slow_kf = df_predict['Kalman_HAM_Slow'].to_numpy()

prob_up, prob_down = [], []
signal_log = []
bar_status = []

# Crossover divergence engine with standard deviations mapping
divergence = fast_kf - slow_kf
rolling_div_std = pd.Series(divergence).rolling(window=10, min_periods=1).std().fillna(1e-6).to_numpy()

for i in range(len(ham_vals)):
    div = divergence[i]
    std_val = rolling_div_std[i] if rolling_div_std[i] > 0 else 1e-6
    
    z_score = div / std_val
    p_up = norm.cdf(z_score)
    p_up = np.clip(p_up, 0.01, 0.99)
    
    prob_up.append(round(p_up, 2))
    prob_down.append(round(1.0 - p_up, 2))
    
    if fast_kf[i] > slow_kf[i]:
        signal_log.append("🟢 BUY (Bullish Cross)")
    else:
        signal_log.append("🔴 SELL (Bearish Cross)")
        
    if i == len(ham_vals) - 1 and is_live_candle_running:
        bar_status.append("🔄 LIVE ACTIVE")
    else:
        bar_status.append("🔒 FROZEN")

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down
df_predict['Signal'] = signal_log
df_predict['Bar_Status'] = bar_status

if is_live_candle_running and len(df_predict) > 0:
    current_sig = df_predict['Signal'].iloc[-1].split(" ")[1]
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
    st.metric(label="Kalman Cross State", value=f"{latest_row['Signal']}")
with col3:
    st.metric(label="Crossover Prob Up", value=f"{latest_row['Prob_Up']*100:.0f}%", delta="Momentum Expanding" if latest_row['Prob_Up'] > 0.5 else "Momentum Contracting")
with col4:
    st.metric(label="Original HAM Value", value=f"{latest_row['Hurst_Amp_Momentum']:.4f}", help="Your precise original HAM output.")

clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Kalman_HAM_Fast', 'Kalman_HAM_Slow', 'Signal', 'Prob_Up', 'Prob_Down', 'Bar_Status']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'BTC Close', 'Hurst_Amp_Momentum': 'Raw HAM', 'Kalman_HAM_Fast': 'Fast Kalman (HAM)', 'Kalman_HAM_Slow': 'Slow Kalman (HAM)'}, inplace=True)

display_df['BTC Close'] = display_df['BTC Close'].round(2)
display_df['Raw HAM'] = display_df['Raw HAM'].round(4)
display_df['Fast Kalman (HAM)'] = display_df['Fast Kalman (HAM)'].round(4)
display_df['Slow Kalman (HAM)'] = display_df['Slow Kalman (HAM)'].round(4)

display_df_inverted = display_df.iloc[::-1].copy()
display_df_inverted.index = display_df_inverted.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Dual Kalman Crossover Matrix (Original Baseline Preserved)")
st.dataframe(display_df_inverted, use_container_width=True, height=750)
