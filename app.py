import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: 2-Year Precision Microstructure Engine")

# 1. Kalman Engine
def apply_kalman(data, q=0.0001, r=2.5):
    if len(data) == 0: return np.array([])
    x = data[0]; p = 100.0; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

# 2. Data Loader
@st.cache_data
def get_data():
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="2y", interval="1h")
    return df.ffill().dropna()

# 3. Execution Pipeline
with st.spinner("Processing Precision Engine..."):
    df = get_data()
    
    if df is not None and len(df) > 50:
        # Core Features
        df['Kalman_Price'] = apply_kalman(df['Close'].values)
        df['c_Combined'] = df['Close'] - df['Kalman_Price']
        
        # Microstructure Features
        df['Order_Imbalance'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Body_Center'] = (df['Open'] + df['Close']) / 2
        df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # --- MULTIPLICATION LOGIC (Fixed Integration) ---
        df['Entry_Quality_Score'] = (
            (df['c_Combined'] * 0.5) +                   
            ((df['Order_Imbalance'] - 0.5) * 100) +      
            ((df['Body_Imbalance'] - 0.5) * 50)          
        )
        
        df['Trade_Signal'] = np.where(df['Entry_Quality_Score'] > 10, "BUY", 
                             np.where(df['Entry_Quality_Score'] < -10, "SELL", "WAIT"))
        # -------------------------------------------------

        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df = df.dropna()
        
        # Split Logic
        split_idx = int(len(df) * 0.50)
        test = df.iloc[split_idx:].sort_index(ascending=False)
        
        st.write(f"✅ Data Ready. Total: {len(df)} Candles | Prediction Window: {len(test)}")
        
        # Final Display
        cols_to_show = ['Close', 'c_Combined', 'Entry_Quality_Score', 'Trade_Signal']
        st.dataframe(test[cols_to_show], use_container_width=True)
    else:
        st.error("Data fetch error. Refresh or check ticker.")
