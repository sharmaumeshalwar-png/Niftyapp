import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Clean Kalman Engine", layout="wide")
st.title("⚡ BTC Live 1-Hour Standalone Breakout Engine")
st.write("🎯 **Clean Setup:** 2-Year Window + 50-Candle Volume Momentum Average Cross Engine")

# =====================================================================
# MATHEMATICAL ENGINE (Flexible Kalman Filter Function)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = q_val      
    r = r_val        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

@st.cache_data(ttl=60)
def pull_historical_data_failsafe():
    """Dual-Node Network: Pulls optimized historical data from YFinance with automatic Kraken fallback"""
    requested_period = "730d" 
    try:
        raw_df = yf.download("BTC-USD", period=requested_period, interval="1h", multi_level_index=False)
        if not raw_df.empty and len(raw_df) > 500:
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = [str(col[0]).upper() for col in raw_df.columns]
            else:
                raw_df.columns = [str(col).upper() for col in raw_df.columns]
                
            df = pd.DataFrame(index=raw_df.index)
            df['Open'] = pd.to_numeric(raw_df['OPEN'].values.flatten(), errors='coerce')
            df['High'] = pd.to_numeric(raw_df['HIGH'].values.flatten(), errors='coerce')
            df['Low'] = pd.to_numeric(raw_df['LOW'].values.flatten(), errors='coerce')
            df['Close'] = pd.to_numeric(raw_df['CLOSE'].values.flatten(), errors='coerce')
            df['Volume'] = pd.to_numeric(raw_df['VOLUME'].values.flatten(), errors='coerce')
            return df
    except Exception:
        pass

    try:
        kraken_url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 60} 
        response = requests.get(kraken_url, params=params, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            data = res_json['result']['XXBTZUSD']
            parsed_data = []
            for item in data:
                parsed_data.append({
                    "Timestamp": pd.to_datetime(item[0], unit='s'),
                    "Open": float(item[1]),
                    "High": float(item[2]),
                    "Low": float(item[3]),
                    "Close": float(item[4]),
                    "Volume": float(item[6])
                })
            df = pd.DataFrame(parsed_data)
            df.set_index("Timestamp", inplace=True)
            return df
    except Exception:
        return pd.DataFrame()

with st.spinner("Processing Matrix Framework..."):
    df_raw = pull_historical_data_failsafe()
    if df_raw.empty:
        st.error("🚨 Both Data Endpoints are unreachable or returned empty frames. Please refresh.")
        st.stop()
        
    df = df_raw.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # VOLUME ENGINE: Dynamic Multiplier (24 Hour Baseline Setup)
    df['Vol_MA_24'] = df['Volume'].rolling(window=24).mean()
    df['Vol_Multiplier'] = df['Volume'] / (df['Vol_MA_24'] + 1e-10)
    df['Vol_Multiplier'] = df['Vol_Multiplier'].clip(lower=0.5, upper=3.0)

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features Space
    df['Sign_Change'] = (np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))).
