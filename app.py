import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="ETH Master Signal Engine", layout="wide")
st.title("⚡ Ethereum (ETH-USD) Pure Action Master Engine")
st.write("🎯 **Pure Direct Signals:** Hurst-Amplified Momentum & Previous Candle Break Validation (100% Locked Core)")

# =====================================================================
# MATHEMATICAL ENGINES (Fixed Loop & Real-Time Safe - 100% UNTOUCHED ORIGINAL)
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
# 🛡️ SYSTEM DATA INGESTION (Strict Ingestion to ETH-USD)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching Live ETH Data..."):
    try:
        df = yf.download(tickers="ETH-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
            # ⛔ STRICT LEAKAGE PROTECTION BLOCK
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
# 🔥 GLOBAL CALCULATION (Exact Original Pipeline - UNTOUCHED VALUES)
# =====================================================================
close_arr = df['Close'].values

# Strict Isolated Price Kalman Baseline Calculation
df['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)

# ATR calculation without lookahead/bfill bias
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(14).mean().ffill() 

# Hurst Vector Generation on full window
df['Hurst'] = calculate_rolling_hurst(close_arr, window=100)

# 🎯 NEW CHRONOLOGY FOR MOMENTUM: Strictly (Close - Kalman_Baseline)
raw_weighted_momentum = df['Close'] - df['Kalman_Baseline']

# Pass the exact raw momentum through untouched custom filter
df['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)

# 🔥 THE MAGICAL MULTIPLICATION (Unaltered Core Scaling)
df['Hurst_Amp_Momentum'] = df['Weighted_Momentum'] * (df['Hurst'] * 2.0)

# Clean NaNs strictly before creating rolling statistical channels
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

# =====================================================================
# 🧠 CANDLE HIGH/LOW BREAK VALIDATION LOGIC
# =====================================================================
validation_signals = []
validation_signals.append("HOLD / NO DATA")

for i in range(1, len(df)):
    current_mom = df['Hurst_Amp_Momentum'].iloc[i]
    prev_mom = df['Hurst_Amp_Momentum'].iloc[i-1]
    
    current_low = df['Low'].iloc[i]
    prev_low = df['Low'].iloc[i-1]
    
    current_high = df['High'].iloc[i]
    prev_high = df['High'].iloc[i-1]
    
    # 🔴 Down Momentum + Price Action Confirmation
    if current_mom < prev_mom and current_low < prev_low:
        status = "🔴 RED SIGNAL"
    # 🟢 Up Momentum + Price Action Confirmation
    elif current_mom > prev_mom and current_high > prev_high:
        status = "🟢 GREEN SIGNAL"
    # ⏳ Whipsaw Trap Caught (No Break)
    else:
        status = "⏳ HOLD"
        
    validation_signals.append(status)

df['Action_Validation'] = validation_signals

# =====================================================================
# 🎛️ DASHBOARD DISPLAY PANEL (Zero Repaint Frozen Layout)
# =====================================================================
df_predict = df.copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} Pure Live Ethereum Candles (100% Untouched Core)!**")

clean_cols = ['Close', 'High', 'Low', 'Hurst_Amp_Momentum', 'Action_Validation', 'Accumulator_Channel']
display_df = df_predict[clean_cols].copy()

display_df.rename(columns={'Close': 'Close_Raw'}, inplace=True)

# Exact formatting specifications
display_df['Hurst_Amp_Momentum'] = display_df['Hurst_Amp_Momentum'].round(4)
for c in ['Close_Raw', 'High', 'Low']:
    display_df[c] = display_df[c].round(2)

# Latest records strictly on top for grid view
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Whipsaw Protection Ethereum Execution Grid")
st.dataframe(display_df, use_container_width=True, height=750)
