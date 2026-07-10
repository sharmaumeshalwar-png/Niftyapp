import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Nifty 50: 2-Year Bulletproof Microstructure Engine")

# 1. Kalman Logic with Empty-Check
def apply_kalman(data, q=0.0001, r=2.5):
    if data is None or len(data) == 0: return np.array([])
    x = data[0]; p = 100.0; filtered = []
    for z in data:
        p += q; k = p / (p + r); x += k * (z - x); p *= (1 - k)
        filtered.append(x)
    return np.array(filtered)

# 2. Robust Data Fetching
@st.cache_data
def get_full_microstructure():
    end = datetime.now()
    start = end - timedelta(days=730)
    # Ticker approach is safer than yf.download
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(start=start, end=end, interval="1h")
    
    if df.empty: return None
    
    df = df.ffill().dropna()
    # Ensure columns exist
    cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in cols): return None
    
    return df[cols].copy()

# 3. Main Engine Execution
with st.spinner("Recovering Historical Nifty Data..."):
    df = get_full_microstructure()
    
    if df is not None and len(df) > 50:
        # Features
        df['Kalman_Price'] = apply_kalman(df['Close'].values)
        df['c_Combined'] = df['Close'] - df['Kalman_Price']
        df['Order_Imbalance'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Body_Center'] = (df['Open'] + df['Close']) / 2
        df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        df = df.dropna()
        
        # 50:50 Split
        split_idx = int(len(df) * 0.50)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:].copy()
        
        # ML
        features = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance']
        model = RandomForestClassifier(n_estimators=50, max_depth=3)
        model.fit(train[features], train['Target'])
        test['Prob'] = model.predict_proba(test[features])[:, 1]
        
        st.write(f"✅ Success! Total Data Points: {len(df)} | Prediction Window: {len(test)}")
        st.dataframe(test[['Close', 'c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Prob']].sort_index(ascending=False), use_container_width=True)
    else:
        st.error("Data source failed or empty. Please check if '^NSEI' ticker is accessible or try changing interval.")
