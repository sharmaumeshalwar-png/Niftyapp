import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Direct Crypto Signals:** 50:50 Split 1-Hour Candle Leak-Free Execution Matrix (IST & Correct HA Fix)")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Backward-Looking / No Leakage)
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

def calculate_rolling_hurst_leak_free(price_series, window=100):
    hurst_values = np.full(len(price_series), 0.5) 
    
    # 🛡️ LEAKAGE PROTECTION STEP: No np.roll wrapping.
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i - window + 1 : i + 1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        
        rs_ratio = r_val / s_val
        if rs_ratio > 0:
            h = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h, 0.0, 1.0)
            
    return hurst_values

# =====================================================================
# 🛠️ CORRECTED HEIKIN-ASHI ENGINE (Explicit Flat Vectors / Zero Leakage)
# =====================================================================
def apply_heikin_ashi(df_in):
    """
    Applies Heikin-Ashi transformation by explicitly flattening Pandas series 
    into 1D numpy arrays to guarantee exact row-by-row mathematical matching.
    """
    # Force alignment by extracting pure 1D numpy arrays to bypass MultiIndex pandas calculation bugs
    op = df_in['Open'].to_numpy().flatten()
    hi = df_in['High'].to_numpy().flatten()
    lo = df_in['Low'].to_numpy().flatten()
    cl = df_in['Close'].to_numpy().flatten()
    
    # 1. Correct HA_Close Vector Calculation
    ha_close = (op + hi + lo + cl) / 4.0
    
    # 2. Sequential HA_Open Calculation
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0
    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    # 3. Dynamic Extremes Boundary Capture
    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))
    
    # Pack cleanly back into index-matched DataFrame
    df_out = df_in.copy()
    df_out['HA_Open'] = ha_open
    df_out['HA_High'] = ha_high
    df_out['HA_Low'] = ha_low
    df_out['HA_Close'] = ha_close
    return df_out

# -----------------------------------------------------------------
# 🛡️ SYSTEM DATA INGESTION (BTC-USD Setup - 2y, 1h)
# -----------------------------------------------------------------
df = None
with st.spinner("Fetching 2-Year Hourly Bitcoin Data from Yahoo Finance..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        
        # Safe MultiIndex Flattening Rule for yfinance columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 120: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            df = df.iloc[:-1] # Live Running Candle Protection
            
            # 🌍 TIMEZONE CONVERSION TO INDIAN STANDARD TIME (IST)
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

# =====================================================================
# ⚡ CORE TRANSFORMATIONS & KINEMATICS
# =====================================================================
# Apply corrected Heikin-Ashi calculation logic first
df = apply_heikin_ashi(df)

# Strict 50:50 split matrix execution
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} IST Bitcoin HA Candles (Leak-Free Engine Active)!**")

# Kinematic Routing based on the fixed HA_Close values
df_predict['Close_Raw'] = df_predict['HA_Close']
close_arr = df_predict['Close_Raw'].values

df_predict['Kalman_Baseline'] = apply_kalman_filter_custom(close_arr, initial_p=50.0, q_val=0.0005, r_val=0.2)
df_predict['Hurst'] = calculate_rolling_hurst_leak_free(close_arr, window=100)

raw_weighted_momentum = df_predict['Close_Raw'] - df_predict['Kalman_Baseline']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(raw_weighted_momentum.values, initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['Hurst_Amp_Momentum'] = df_predict['Weighted_Momentum'] * (df_predict['Hurst'] * 2.0)

df_predict.dropna(subset=['Hurst'], inplace=True)

# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    'HA_Open',
    'HA_High',
    'HA_Low',
    'HA_Close', 
    'Hurst', 
    'Weighted_Momentum', 
    'Hurst_Amp_Momentum'
]
display_df = df_predict[clean_cols].copy()

for c in clean_cols:
    display_df[c] = display_df[c].round(2)

# Sort descending for active trading view (Latest IST candles on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.subheader("📋 50:50 Split Pure Kinematic Analysis Matrix (IST - Heikin-Ashi Corrected)")
st.dataframe(display_df, use_container_width=True, height=650)
