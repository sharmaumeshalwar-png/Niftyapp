import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: Pattern Recognition Engine")

@st.cache_data(ttl=3600)
def get_geometric_data():
    # Last 2 years
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Geometry Features: Price Range aur Volatility
    df['Range_10'] = df['Price'].rolling(10).max() - df['Price'].rolling(10).min()
    df['Std_Dev'] = df['Price'].rolling(20).std()
    df['Skew'] = df['Price'].rolling(20).skew()
    
    return df.dropna()

data = get_geometric_data()

# Model: Isolation Forest (Ye market ke "Anomaly" ya "Big Breakout" patterns dhundta hai)
model = IsolationForest(n_estimators=200, contamination=0.05)
model.fit(data)

# Anomaly Score: -1 (Outlier/Big Move) to 1 (Normal)
data['Anomaly_Score'] = model.decision_function(data)
data['Signal'] = model.predict(data) # -1 means pattern shift

st.subheader("📋 Pattern Discovery Audit")
st.write("Jahan Signal '-1' hai, wahan market ka 'Pattern' badal raha hai (Breakout/Reversal).")
st.dataframe(data.sort_index(ascending=False).head(20), use_container_width=True)
