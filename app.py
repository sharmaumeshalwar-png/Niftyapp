import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Smart-Feature Engine")

@st.cache_data(ttl=3600)
def get_optimized_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # "Smart Features" - Market ke har pehlu ko cover karne ke liye
    # 1. Trend: SMA 50 vs 200
    df['Trend'] = df['Price'].rolling(50).mean() / df['Price'].rolling(200).mean()
    # 2. Volatility: Standard Deviation
    df['Vol'] = df['Price'].rolling(50).std()
    # 3. Momentum: RSI-style acceleration
    df['Acc'] = df['Price'].diff(10).rolling(10).mean()
    # 4. Cycle: Price vs SMA ratio
    df['Cycle'] = df['Price'] / df['Price'].rolling(100).mean()
    
    df['Future_Move'] = df['Price'].pct_change(150).shift(-150)
    return df.dropna()

df = get_optimized_data()

# Model: 500 trees are enough with "Smart Features"
model = RandomForestRegressor(n_estimators=500, max_depth=12, n_jobs=-1).fit(
    df[['Trend', 'Vol', 'Acc', 'Cycle']], df['Future_Move']
)

df['Pred_Move'] = model.predict(df[['Trend', 'Vol', 'Acc', 'Cycle']])
df['Target'] = df['Price'] * (1 + df['Pred_Move'])
df['Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Smart Discovery Audit")
st.dataframe(df.sort_index(ascending=False), use_container_width=True)
