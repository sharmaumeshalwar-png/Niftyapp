import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Data Recovery Engine")

@st.cache_data(ttl=3600)
def fetch_safe_data():
    try:
        # Ticker check: Kabhi kabhi sirf '^NSEI' ya 'NSEI.NS' kaam karta hai
        data = yf.download("^NSEI", period="1y", interval="1h", progress=False)
        
        if data.empty:
            return None
        
        # Data cleaning
        df = data[['Close']].copy()
        df.columns = ['Price']
        df = df.ffill().dropna()
        
        # Features check
        df['SMA_150'] = df['Price'].rolling(150).mean()
        df['Volatility'] = df['Price'].rolling(150).std()
        
        # Drop NaN after rolling
        df = df.dropna()
        
        return df
    except Exception as e:
        return str(e)

data_result = fetch_safe_data()

if isinstance(data_result, str):
    st.error(f"Data Fetch Error: {data_result}")
elif data_result is None:
    st.error("No data found for Nifty 50. Server response empty.")
else:
    st.success(f"Data successful! Loaded {len(data_result)} records.")
    
    # ML Prediction
    data_result['Dynamic_Target'] = data_result['Price'] + (data_result['Price'] - data_result['SMA_150']) * 0.5
    X = data_result[['Price', 'SMA_150', 'Volatility']]
    y = data_result['Dynamic_Target']
    
    model = RandomForestRegressor(n_estimators=50).fit(X, y)
    data_result['Prediction'] = model.predict(X)
    
    st.subheader("📋 Discovery Table")
    st.data_editor(data_result.sort_index(ascending=False).head(20), use_container_width=True)
