import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Liquidity Decoder", layout="wide")

st.title("🎯 Nifty 50: Liquidity Hunt Engine")

@st.cache_data(ttl=3600)
def get_hunt_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    prev_low = df['Low'].shift(1).rolling(5).min().values
    prev_high = df['High'].shift(1).rolling(5).max().values
    
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_hunt_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

# Streamlit UI Components
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Model Status")
    latest = test.iloc[-1]
    prob = model.predict_proba(latest[features].values.reshape(1, -1))[0][1]
    
    st.metric("Hunt Probability", f"{prob:.2%}")
    if prob > 0.6:
        st.error("🚨 TRAP CONFIRMED")
    else:
        st.success("✅ NORMAL MARKET")

with col2:
    st.subheader("Live Hunt Audit (2-Year Split)")
    st.dataframe(test.sort_index(ascending=False).head(10), use_container_width=True)
