import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty Master Kinematics Engine", layout="wide")
st.title("⚡ Nifty 50 Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Non-Linear Hurst Kinematics & 5-Channel Accumulator (100% Leak-Proof)")

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
# 🛡️ SYSTEM DATA INGESTION (Nifty 50 Index Setup - 2y, 1h)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching 2-Year Hourly Nifty 50 Data..."):
    try:
        df = yf.download(tickers="^NSEI", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            # Live Running Candle Protection (100% Leak-Proof Guard)
            df = df.iloc[:-1]
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient data lines from Yahoo Finance.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# 🔥 CRITICAL DATA SCIENCE RULE: 50:50 split FIRST to prevent lookahead leakage globally
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Nifty Candles (No Leakage)!**")

# =====================================================================
# ⚡ THE SUPREME MODIFICATION: NON-LINEAR HURST KINEMATICS CORE
# =====================================================================
df_predict['Close_Raw'] = df_predict['Close']
close_arr = df_predict['Close_Raw'].values

# 1. Base Kalman System & Innovation Error Extraction
df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Kalman_Innovation'] = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']

# 2. Pure Hurst Base Calculation
df_predict['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 3. MODIFICATION 1: HURST VELOCITY ENGINE (Allag Column ke liye)
df_predict['Hurst_Velocity'] = df_predict['Hurst'].diff(5).fillna(0.0)
df_predict['Hurst_Acceleration'] = np.exp(df_predict['Hurst_Velocity'] * 3.0)

# 4. MODIFICATION 2: KALMAN ERROR ENERGY COUPLING (Shock Index Column)
df_predict['Error_Energy'] = df_predict['Kalman_Innovation'].rolling(window=20, min_periods=1).std().fillna(1.0)
df_predict['Shock_Index'] = np.abs(df_predict['Kalman_Innovation']) / (df_predict['Error_Energy'] + 1e-10)

# 5. Raw Price-based Weighted Momentum
raw_weighted_momentum = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 6. ✨ THE MAGICAL CONVERGENCE LAYER
df_predict['Hurst_Amp_Momentum'] = (
    df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0) * df_predict['Hurst_Acceleration'] * (1.0 + df_predict['Shock_Index'] * 0.5)
)

# Clean NaNs strictly before creating rolling statistical channels
df_predict.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 📊 1 TO 5 CHANNEL ACCUMULATOR ENGINE
# =====================================================================
mom_vals = df_predict['Hurst_Amp_Momentum'].to_numpy()
rolling_window = 50
mom_mean = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).mean().to_numpy()
mom_std = df_predict['Hurst_Amp_Momentum'].rolling(window=rolling_window, min_periods=1).std().fillna(1.0).to_numpy()

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

df_predict['Raw_Channel'] = channels
df_predict['Accumulator_Channel'] = accumulator

# 🤖 SIGNAL GENERATION
signal_log = []
current_sig = "🔴 SELL"  

for i in range(len(df_predict)):
    acc_chan = accumulator[i]
    if acc_chan >= 4:
        current_sig = "🟢 BUY"
    elif acc_chan <= 2:
        current_sig = "🔴 SELL"
    signal_log.append(current_sig)

df_predict['Signal'] = signal_log

# 🚀 PROBABILITY MATRIX
prob_up = []
prob_down = []

for i in range(len(df_predict)):
    acc_chan = accumulator[i]
    sig = signal_log[i]
    
    if acc_chan == 5:
        p_up = 0.95
    elif acc_chan == 4:
        p_up = 0.75
    elif acc_chan == 3:
        p_up = 0.55 if sig == "🟢 BUY" else 0.45
    elif acc_chan == 2:
        p_up = 0.25
    else: 
        p_up = 0.05
        
    prob_up.append(round(p_up, 2))
    prob_down.append(round(1.0 - p_up, 2))

df_predict['Prob_Up'] = prob_up
df_predict['Prob_Down'] = prob_down

# =====================================================================
# 📋 MATRIX FORMATTING (Naye Columns Add Kar Diye Hain)
# =====================================================================
clean_cols = [
    'Close_Raw', 
    'Hurst', 
    'Hurst_Acceleration', 
    'Shock_Index', 
    'Hurst_Amp_Momentum', 
    'Accumulator_Channel', 
    'Signal', 
    'Prob_Up', 
    'Prob_Down'
]
display_df = df_predict[clean_cols].copy()

# Rounding off for neat display
for c in ['Close_Raw', 'Hurst', 'Hurst_Acceleration', 'Shock_Index', 'Hurst_Amp_Momentum']:
    display_df[c] = display_df[c].round(4 if c != 'Close_Raw' else 2)

# Reverse index for latest on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 5-Channel Kinematic Nifty 50 Action Matrix")
st.dataframe(display_df, use_container_width=True, height=750)
