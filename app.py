import streamlit as st
import numpy as np
import pandas as pd
import requests

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC/USDT) Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Direct Crypto Signals:** Dual H.A.M. Matrix with Binance Original Volume in IST")

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

def apply_heikin_ashi(df_in):
    op = df_in['Open'].to_numpy().flatten()
    hi = df_in['High'].to_numpy().flatten()
    lo = df_in['Low'].to_numpy().flatten()
    cl = df_in['Close'].to_numpy().flatten()
    
    ha_close = (op + hi + lo + cl) / 4.0
    
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0
    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))
    
    df_out = df_in.copy()
    df_out['HA_Open'] = ha_open
    df_out['HA_High'] = ha_high
    df_out['HA_Low'] = ha_low
    df_out['HA_Close'] = ha_close
    return df_out

# =====================================================================
# 🛡️ SYSTEM DATA INGESTION (Binance Original Data Direct Fetch)
# =====================================================================
@st.cache_data(ttl=300)
def fetch_binance_original_data():
    # Fetch directly 1000 candles from Binance US/Global Spot API
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data, columns=[
                'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_Time', 'Quote_Volume', 'Trades', 'Taker_Base', 'Taker_Quote', 'Ignore'
            ])
            df['Timestamp'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
            df.set_index('Timestamp', inplace=True)
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        pass
    
    # Fallback to Binance US Endpoint if main endpoint blocks IP
    url_us = "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000"
    try:
        res_us = requests.get(url_us, headers=headers, timeout=10)
        data_us = res_us.json()
        if isinstance(data_us, list) and len(data_us) > 0:
            df = pd.DataFrame(data_us, columns=[
                'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_Time', 'Quote_Volume', 'Trades', 'Taker_Base', 'Taker_Quote', 'Ignore'
            ])
            df['Timestamp'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
            df.set_index('Timestamp', inplace=True)
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        pass

    return pd.DataFrame()

df = None
with st.spinner("Fetching Original Binance BTC/USDT Real-Time Data..."):
    df = fetch_binance_original_data()
    
if df.empty or len(df) < 100:
    st.error("🚨 API IP Rate Limited. Kripya 1-2 minute mein page refresh karein!")
    st.stop()

# Convert Index to Indian Standard Time (IST)
df.index = df.index.tz_convert('Asia/Kolkata')

# Drop active incomplete running candle
df = df.iloc[:-1]

# =====================================================================
# ⚡ CORE TRANSFORMATIONS & DUAL KINEMATICS ENGINE
# =====================================================================
df = apply_heikin_ashi(df)

# Pure Original Volume Factor Logic (Current Volume / 20-period Moving Average)
vol_ma20 = df['Volume'].rolling(window=20).mean()
df['Volume_Factor'] = (df['Volume'] / vol_ma20).fillna(1.0)

# 50:50 Split Matrix Execution
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced {len(df_predict)} IST Candles with Original Binance Spot Volume!**")

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close = df_predict['Close'].to_numpy().flatten()
df_predict['Hurst_Normal'] = calculate_rolling_hurst_leak_free(normal_close, window=100)
kalman_base_normal = apply_kalman_filter_custom(normal_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal = apply_kalman_filter_custom(normal_close - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Host_x_Vol_Normal'] = df_predict['Hurst_Normal'] * df_predict['Volume_Factor']
df_predict['HAM_Normal'] = np.array(momentum_normal) * df_predict['Host_x_Vol_Normal']

# --- PATH B: HEIKIN-ASHI CANDLE KINEMATICS ---
ha_close = df_predict['HA_Close'].to_numpy().flatten()
df_predict['Hurst_HA'] = calculate_rolling_hurst_leak_free(ha_close, window=100)
kalman_base_ha = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha = apply_kalman_filter_custom(ha_close - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Host_x_Vol_HA'] = df_predict['Hurst_HA'] * df_predict['Volume_Factor']
df_predict['HAM_HeikinAshi'] = np.array(momentum_ha) * df_predict['Host_x_Vol_HA']

df_predict.dropna(subset=['Hurst_Normal', 'Hurst_HA'], inplace=True)

# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    'Close',              
    'HA_Close',           
    'Volume',             # Pure Binance Volume Column
    'Volume_Factor',      
    'Hurst_Normal',       
    'Host_x_Vol_Normal',  
    'HAM_Normal',         
    'HAM_HeikinAshi'      
]
display_df = df_predict[clean_cols].copy()

for c in ['Close', 'HA_Close', 'Volume_Factor', 'Hurst_Normal', 'Host_x_Vol_Normal', 'HAM_Normal', 'HAM_HeikinAshi']:
    display_df[c] = display_df[c].round(2)

display_df['Volume'] = display_df['Volume'].round(3)

# Latest IST candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.subheader("📋 Pure Binance Volume & Kinematic Matrix")
st.dataframe(display_df, use_container_width=True, height=650)
