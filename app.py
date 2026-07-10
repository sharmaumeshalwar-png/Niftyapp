import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: 2-Year Microstructure Engine (Original Logic)")

# Kalman Engine
def apply_kalman(data, q=0.0001, r=2.5):
    x = data[0]; p = 100.0; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

# Data Fetch & Microstructure Logic
@st.cache_data
def get_full_microstructure():
    end = datetime.now()
    start = end - timedelta(days=730)
    df = yf.download("^NSEI", start=start, end=end, interval="1h", auto_adjust=True)
    df = df.ffill().dropna()
    
    # 1. Kalman Price
    df['Kalman_Price'] = apply_kalman(df['Close'].values)
    
    # 2. Weighted Momentum (Original Logic)
    df['c_Combined'] = df['Close'] - df['Kalman_Price']
    
    # 3. Microstructure Features
    df['Order_Imbalance'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # 4. Target
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df.dropna()

with st.spinner("Calculating Original Microstructure Features..."):
    df = get_full_microstructure()
    
    # ML Features
    features = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance']
    
    # 50:50 Split
    split_idx = int(len(df) * 0.50)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:].copy()
    
    # Original ML Logic
    model = RandomForestClassifier(n_estimators=100, max_depth=4)
    model.fit(train[features], train['Target'])
    test['Prob'] = model.predict_proba(test[features])[:, 1]
    
    st.write(f"### Total Candles: {len(df)} | Prediction Window (Last 50%): {len(test)}")
    st.dataframe(test[['Close', 'c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Prob']].sort_index(ascending=False), use_container_width=True)
