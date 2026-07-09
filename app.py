import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Stable Discovery [No-Crash Mode]")

@st.cache_data(ttl=60)
def get_data():
    # 2 saal ka data, 1 hour interval
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Simple Features: Previous 1 hour and Previous 150 hours SMA
    df['Prev_Hour'] = df['Price'].shift(1)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    
    # Target: Next hour ka price
    df['Target'] = df['Price'].shift(-1)
    return df.dropna()

data = get_data()

# Model
X = data[['Prev_Hour', 'SMA_150']]
y = data['Target']

model = LinearRegression()
model.fit(X, y)

# Prediction
data['Prediction'] = model.predict(X)
# 150 candles (hours) aage ka projection
data['Future_Target'] = data['Prediction'] * 1.05 

st.subheader("📋 Latest Nifty Status (July 2026)")
st.dataframe(data.sort_index(ascending=False).head(20), use_container_width=True)
