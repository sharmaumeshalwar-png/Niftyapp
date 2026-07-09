import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Smart-Ensemble Engine [100-Model Cluster]")

@st.cache_data(ttl=3600)
def get_smart_features():
    raw = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # 100-Model ke liye multi-dimensional features
    # Trend Aspect
    df['SMA_50_200'] = df['Price'].rolling(50).mean() / df['Price'].rolling(200).mean()
    # Volatility Aspect
    df['Vol_20'] = df['Price'].rolling(20).std()
    # Momentum Aspect
    df['RSI'] = df['Price'].diff(14).rolling(14).mean()
    # Cycle Aspect
    df['Mean_Dist'] = (df['Price'] - df['Price'].rolling(100).mean()) / df['Price']
    
    # Target: 150 candles ka ACTUAL move
    df['Target'] = df['Price'].pct_change(150).shift(-150)
    return df.dropna()

data = get_smart_features()

# Ensemble Strategy: 100 models, har ek model thode alag features aur subset par train hoga
predictions = []
for i in range(100):
    # Har model ke liye random feature subset aur data sampling
    subset = data.sample(frac=0.8)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1)
    model.fit(subset[['SMA_50_200', 'Vol_20', 'RSI', 'Mean_Dist']], subset['Target'])
    
    # Live market prediction
    pred = model.predict(data[['SMA_50_200', 'Vol_20', 'RSI', 'Mean_Dist']])
    predictions.append(pred)

# Final Prediction: 100 models ka average (Ensemble Voting)
data['Smart_Prediction'] = np.mean(predictions, axis=0)
data['Target_Price'] = data['Price'] * (1 + data['Smart_Prediction'])

st.subheader("📋 100-Model Ensemble Discovery")
st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)
