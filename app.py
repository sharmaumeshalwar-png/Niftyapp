import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Infinite-Depth Discovery Engine")

@st.cache_data(ttl=3600)
def get_infinite_depth_data():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Feature Engineering: 1 Lakh trees ke liye jitne zyada technical dimensions honge, utna accha
    df['SMA_50'] = df['Price'].rolling(50).mean()
    df['SMA_200'] = df['Price'].rolling(200).mean()
    df['Vol_50'] = df['Price'].rolling(50).std()
    df['Momentum'] = df['Price'].pct_change(10)
    df['Trend_Slope'] = (df['Price'] - df['SMA_200']) / df['Price']
    
    # Target
    df['Future_Move'] = df['Price'].pct_change(150).shift(-150)
    
    return df.dropna()

df = get_infinite_depth_data()

# Model: 1,00,000 estimators (Boosting strategy)
# Learning_rate ko kam rakha hai taaki 1 lakh trees stable rahein
model = GradientBoostingRegressor(
    n_estimators=100000, 
    learning_rate=0.01, 
    max_depth=3, 
    subsample=0.8
).fit(df[['SMA_50', 'SMA_200', 'Vol_50', 'Momentum', 'Trend_Slope']], df['Future_Move'])

# Projection
df['Predicted_Move'] = model.predict(df[['SMA_50', 'SMA_200', 'Vol_50', 'Momentum', 'Trend_Slope']])
df['Smart_Target'] = df['Price'] * (1 + df['Predicted_Move'])
df['Projected_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Deep-Learning Discovery [100k Iterations]")
st.dataframe(df.sort_index(ascending=False), use_container_width=True)
