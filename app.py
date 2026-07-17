import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Page Configuration
st.set_page_config(page_title="BTC Range Bar Signal Engine", layout="wide")
st.title("⚡ BTC 200-Point Range Bar Master Engine")
st.write("🎯 **Pure Price Action:** Time-Independent 200-Point Range Candles with HAM Probabilities (100% Noise Filtered)")

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
    # Reduced window to 50 for range bars as they compress time series
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
with st.spinner("Fetching Live Bitcoin Data (2 Years, 1-Hour resolution)..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        
        # Robust multi-index column flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1]  # Live Incomplete running candle protection
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
# 🧱 200-POINT RANGE CANDLE GENERATION ENGINE (True Range Bars)
# =====================================================================
# Here we filter the hourly time-series into custom bars that only close 
# when price moves by at least 200 points.
raw_closes = df['Close'].to_numpy(dtype=float)
raw_times = df.index

range_size = 200.0
range_closes = []
range_times = []

# Initialize with the first price point
current_anchor = raw_closes[0]
range_closes.append(current_anchor)
range_times.append(raw_times[0])

for i in range(1, len(raw_closes)):
    price_diff = raw_closes[i] - current_anchor
    
    # If price moves up or down by 200 points or more
    if abs(price_diff) >= range_size:
        # Number of 200-point blocks completed in this move
        num_bars = int(abs(price_diff) // range_size)
        direction = np.sign(price_diff)
        
        for _ in range(num_bars):
            current_anchor += direction * range_size
            range_closes.append(current_anchor)
            range_times.append(raw_times[i])

# Create Range Bar DataFrame
df_range = pd.DataFrame(index=range_times)
df_range['Close'] = range_closes

# 🔥 50:50 Out-of-Sample Split on these Range Bars
split_idx = int(len(df_range) * 0.50)
df_predict = df_range.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Processed {len(df_predict)} Pure 200-Point Range Bars (50% Out-of-Sample)!**")

# Setup Isolated Price Array based on Range Bar closes
close_arr = df_predict['Close'].to_numpy(dtype=float)

# =====================================================================
# 📊 DOWNSTREAM SIGNAL CALCULATIONS ON RANGE BARS
# =====================================================================
# Kalman Baseline on range close
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# Hurst Vector calculation
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=50)

# Exact Weighted Momentum on Range Bars
raw_weighted_momentum = df_predict['Close'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.to_numpy(), initial_p=0.50, q_val=0.001, r_val=0.1)

# Raw HAM (Hurst-Amplified Momentum)
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

# Clean and align
df_predict.dropna(subset=['Hurst', 'Close'], inplace=True)
df_predict = df_predict.ffill().bfill()

# =====================================================================
# 📊 PURE HAM-BASED PROBABILITY ENGINE (No Static Channels)
# =====================================================================
rolling_window = 30 # Slightly tighter window since noise is already filtered
mom_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
mom_mean = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).mean().to_numpy()
mom_std = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).std().fillna(1e-6).to_numpy()
mom_std = np.where(mom_std == 0, 1e-6, mom_std)

prob_up, prob_down = [], []
signal_log = []

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

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down
df_predict['Signal'] = signal_log

# =====================================================================
# 📋 DASHBOARD METRICS & TABLE ONLY
# =====================================================================
latest_row = df_predict.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="BTC Range Close (USD)", 
        value=f"${latest_row['Close']:.2f}", 
        delta=f"${(latest_row['Close'] - df_predict['Close'].iloc[-2]):.2f}",
        help="Updates only when price prints a new 200-Point block."
    )
with col2:
    st.metric(
        label="Active HAM Signal", 
        value=f"{latest_row['Signal']}",
        delta=f"HAM Val: {latest_row['Hurst_Amp_Momentum']:.4f}"
    )
with col3:
    st.metric(
        label="HAM Up Probability", 
        value=f"{latest_row['Prob_Up']*100:.0f}%",
        delta="Bullish Momentum" if latest_row['Prob_Up'] > 0.5 else "Bearish Momentum"
    )
with col4:
    st.metric(
        label="Hurst Intensity (Range)", 
        value=f"{latest_row['Hurst']:.2f}",
        delta="Persistent Trend" if latest_row['Hurst'] > 0.5 else "Mean Reverting"
    )

# Matrix Data Display
clean_cols = ['Close', 'Hurst_Amp_Momentum', 'Signal', 'Prob_Up', 'Prob_Down']
display_df = df_predict[clean_cols].copy()
display_df.rename(columns={'Close': 'Range Close (200-Pt steps)', 'Hurst_Amp_Momentum': 'Raw HAM'}, inplace=True)

# Round values
display_df['Range Close (200-Pt steps)'] = display_df['Range Close (200-Pt steps)'].round(2)
display_df['Raw HAM'] = display_df['Raw HAM'].round(4)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 200-Point Range Bar Signal Matrix (No Time Noise)")
st.dataframe(display_df, use_container_width=True, height=750)
