import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Bulletproof Mode]")

@st.cache_data(ttl=3600)
def get_clean_data():
    raw = yf.download("^NSEI.NS", period="2y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Volatility'] = df['Price'].rolling(150).std()
    
    # Statistical Baseline Target (No shift/lag)
    df['Dynamic_Target'] = df['Price'] + (df['Price'] - df['SMA_150']) * 0.5
    
    # 8-Step Data Cleaning (Crucial to stop ValueError)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

df = get_clean_data()

# ML Pipeline
if df.empty:
    st.error("Data processing failed: No valid numerical data.")
else:
    X = df[['Price', 'SMA_150', 'Volatility']]
    y = df['Dynamic_Target']

    # Model fit
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(X, y)
    
    # Projection
    df['Prediction'] = model.predict(X)
    df['Target_Date'] = df.index + pd.offsets.BusinessDay(23)
    
    st.subheader("📋 Discovery Table (Cleaned & Lag-Free)")
    st.data_editor(
        df.sort_index(ascending=False).head(50),
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
            "Prediction": st.column_config.NumberColumn("ML Projection", format="%.2f"),
            "Dynamic_Target": st.column_config.NumberColumn("Stat. Baseline", format="%.2f"),
        }
    )
