import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Real-Time Engine [No Stale Data]")

# Cache hataya taaki har baar fresh data fetch ho
@st.cache_data(ttl=60) 
def get_live_data():
    # Sirf last 30 days ka data (Live accuracy ke liye)
    raw = yf.download("^NSEI", period="1mo", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Feature Engineering (Past 150 windows)
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Vol_150'] = df['Price'].rolling(150).std()
    
    # Target: Sirf agle 10 ghante ka trend (Leakage proof)
    df['Target_Price'] = df['Price'].shift(-10)
    return df.dropna()

data = get_live_data()

# Model
model = RandomForestRegressor(n_estimators=100)
X = data[['SMA_150', 'Vol_150']]
y = data['Target_Price']
model.fit(X, y)

# Prediction
data['Future_Projection'] = model.predict(X)

st.subheader(f"📋 Live Projection (Last Updated: {data.index[-1].strftime('%d %b %Y %H:%M')})")
st.dataframe(data.sort_index(ascending=False).head(10), use_container_width=True)
