import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Linear Smooth Distance Kalman (Price) + Probability Flow Tracker")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR FILTER
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.0001     
    r = 2.5        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(float(x))
    return filtered_values

# =====================================================================
# MATHEMATICAL ENGINE 2: NON-LINEAR FILTER
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = 1.0  
    q = 0.05   
    r = 0.2    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(float(x))
    return filtered_values

with st.spinner("Aligning 25-Candle Dual Kalman Nifty Microstructure Matrices..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h", progress=False)
        
    if raw_df.empty:
        st.error("🚨 YFinance API Timeout or Indian Market Closed. Please refresh the dashboard.")
        st.stop()
        
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')

    df['Prev_High'] = df['High'].shift(1).ffill().bfill()
    df['Prev_Low'] = df['Low'].shift(1).ffill().bfill()

    df['a_Close'] = df['Close'].astype(float)
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].to_numpy(), initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_matrix + ['Target']).copy()

# =====================================================================
# HARDENED SPLIT AND ERROR
