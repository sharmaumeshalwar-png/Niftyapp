import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: Max 2-Year Professional Engine [Bypass Mode]")

# 1. Kalman Logic
def apply_kalman(data):
    if len(data) == 0: return np.array([])
    x = data[0]; p = 100.0; q = 0.0001; r = 2.5; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

# 2. Data Fetcher with Fallback
@st.cache_data
def get_data_force():
    # Attempt 1: Ticker Object
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="2y", interval="1h")
    
    # Attempt 2: If 1h fails, try 1d (Fallback)
    if df.empty or len(df) < 10:
        df = ticker.history(period="2y", interval="1d")
        st.warning("⚠️ 1h interval restricted by Yahoo. Switching to 1d (Daily) interval.")
        
    return df.ffill().dropna() if not df.empty else None

with st.spinner("Forcing connection to Nifty 50..."):
    df = get_data_force()
    
    if df is not None:
        # Microstructure Logic
        df['Kalman_Price'] = apply_kalman(df['Close'].values)
        df['c_Combined'] = df['Close'] - df['Kalman_Price']
        df['Order_Imbalance'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Body_Center'] = (df['Open'] + df['Close']) / 2
        df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        df = df.dropna()
        
        # 50:50 Split
        split_idx = int(len(df) * 0.50)
        test = df.iloc[split_idx:].sort_index(ascending=False)
        
        st.write(f"✅ Data Loaded. Total Candles: {len(df)} | Displaying Window: {len(test)}")
        st.dataframe(test[['Close', 'c_Combined', 'Order_Imbalance', 'Body_Imbalance']], use_container_width=True)
    else:
        st.error("Data Fetch Failed. Yahoo API may be down for ^NSEI in your region.")
