import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Time-Loop Discovery Engine")

@st.cache_data(ttl=3600)
def get_time_loop_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # LAYER 1: Past 150 Candles ka "Context" (Aaj ka price kyun hai)
    # Humein 150 ghante ka trend aur volatility chahiye
    df['Past_150_Trend'] = df['Price'].rolling(150).mean()
    df['Past_150_Vol'] = df['Price'].rolling(150).std()
    
    # LAYER 2: Aaj ki "Dynamic State" (Aaj ka candle action)
    df['Current_Momentum'] = df['Price'].pct_change(5)
    
    # TARGET: Future 150 candles ka "Net Impact"
    # Hum 'Future_150_Shift' ko sirf training label bana rahe hain, feature nahi
    df['Future_150_Impact'] = df['Price'].shift(-150) / df['Price']
    
    return df.dropna()

data = get_time_loop_data()

# Training: Model seekhega ki 'Past' + 'Current' milkar 'Future' mein kya impact dalte hain
X = data[['Past_150_Trend', 'Past_150_Vol', 'Current_Momentum']]
y = data['Future_150_Impact']

# 100-Model Cluster (Memory-Efficient)
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
model.fit(X, y)

# Prediction
data['Predicted_Future_Impact'] = model.predict(X)
data['Smart_Discovery_Target'] = data['Price'] * data['Predicted_Future_Impact']

st.subheader("📋 Time-Loop Intelligence Audit")
st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)
