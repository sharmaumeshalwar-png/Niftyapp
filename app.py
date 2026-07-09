import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Ultra-Stable Discovery")

@st.cache_data(ttl=3600)
def get_stable_data():
    # Basic data fetch
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Feature Engineering (Smart logic)
    df['SMA_Ratio'] = df['Price'].rolling(50).mean() / df['Price'].rolling(200).mean()
    df['Volatility'] = df['Price'].rolling(30).std()
    df['Momentum'] = df['Price'].pct_change(10)
    
    # Target (Past 150 hours move)
    df['Target'] = df['Price'].pct_change(150).shift(-150)
    return df.dropna()

# Data load check
data = get_stable_data()

if data.empty:
    st.error("Data load nahi ho pa raha. Please check connection.")
else:
    # 500 Trees: Ye model stable hai aur memory nahi khata
    model = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1)
    
    X = data[['SMA_Ratio', 'Volatility', 'Momentum']]
    y = data['Target']
    
    model.fit(X, y)
    
    data['Predicted_Move'] = model.predict(X)
    data['Smart_Target'] = data['Price'] * (1 + data['Predicted_Move'])
    
    st.subheader("📋 Discovery Engine [Stable Mode]")
    st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)
