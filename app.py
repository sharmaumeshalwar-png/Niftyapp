import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 50:50 Backtest Engine")

@st.cache_data(ttl=3600)
def get_backtest_data():
    # 2 saal ka data load karo
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Features
    df['Prev_Hour'] = df['Price'].shift(1)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    
    # Target
    df['Target'] = df['Price'].shift(-1)
    return df.dropna()

data = get_backtest_data()

# 50:50 Split Logic (Total data ka 50% train, 50% test)
split_idx = int(len(data) * 0.5)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

# Training
model = LinearRegression()
model.fit(train[['Prev_Hour', 'SMA_150']], train['Target'])

# Prediction on Testing data (Ye wo 1 saal hai jo model ne pehle nahi dekha)
test = test.copy()
test['Prediction'] = model.predict(test[['Prev_Hour', 'SMA_150']])

st.subheader("📋 50:50 Split Historical Backtest (Last 1 Year Performance)")
st.dataframe(test.sort_index(ascending=False), use_container_width=True)
