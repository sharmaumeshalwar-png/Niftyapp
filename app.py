import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Discovery Engine [Date-Projected]")

@st.cache_data(ttl=3600)
def get_final_data():
    raw = yf.download("^NSEI", period="1y", interval="1h", progress=False)
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close'].ffill().squeeze()
    
    # Features
    df['SMA_150'] = df['Price'].rolling(150).mean()
    df['Volatility'] = df['Price'].rolling(150).std()
    
    # Target: Statistical projection (150 candles/23 days ahead)
    df['Dynamic_Target'] = df['Price'] + (df['Price'] - df['SMA_150']) * 0.5
    
    # Cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = get_final_data()

# Model training
X = df[['Price', 'SMA_150', 'Volatility']]
y = df['Dynamic_Target']
model = RandomForestRegressor(n_estimators=50).fit(X, y)

# Prediction & Projection Date
df['Prediction'] = model.predict(X)
# 150 hours = 23 Business Days
df['Projected_Date'] = df.index + pd.offsets.BusinessDay(23)

st.subheader("📋 Discovery Table with Projected Dates")

# Full view
st.data_editor(
    df.sort_index(ascending=False), 
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
        "Prediction": st.column_config.NumberColumn("ML Target", format="%.2f"),
        "Projected_Date": st.column_config.DateColumn("Date 150 Candles Ahead", format="DD/MM/YYYY"),
    }
)
