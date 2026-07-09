import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: Pattern Recognition Engine [Fixed]")

@st.cache_data(ttl=3600)
def get_geometric_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Feature Engineering
    df['Range_10'] = df['Price'].rolling(10).max() - df['Price'].rolling(10).min()
    df['Std_Dev'] = df['Price'].rolling(20).std()
    df['Skew'] = df['Price'].rolling(20).skew()
    
    return df.dropna()

data = get_geometric_data()
features = ['Range_10', 'Std_Dev', 'Skew']

# Model Training
model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
# Sirf features ko fit karo, Price column ko nahi
model.fit(data[features])

# Prediction
data['Anomaly_Score'] = model.decision_function(data[features])
data['Signal'] = model.predict(data[features]) 

st.subheader("📋 Pattern Discovery Audit")
st.write("Signal -1 = Pattern Shift / Breakout point")
st.dataframe(data.sort_index(ascending=False).head(20), use_container_width=True)
