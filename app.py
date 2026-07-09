import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: 2-Year Pattern Audit (50:50 Split)")

@st.cache_data(ttl=3600)
def get_geometric_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Geometric Patterns
    df['Range_10'] = df['Price'].rolling(10).max() - df['Price'].rolling(10).min()
    df['Std_Dev'] = df['Price'].rolling(20).std()
    df['Skew'] = df['Price'].rolling(20).skew()
    
    return df.dropna()

data = get_geometric_data()
features = ['Range_10', 'Std_Dev', 'Skew']

# 50:50 Split (1 Year Train / 1 Year Test)
split = int(len(data) * 0.5)
train = data.iloc[:split]
test = data.iloc[split:]

# Training on the first 50%
model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
model.fit(train[features])

# Testing/Auditing on the second 50%
test = test.copy()
test['Anomaly_Score'] = model.decision_function(test[features])
test['Signal'] = model.predict(test[features]) # -1 = Pattern Shift

st.subheader("📋 2-Year Audit Report")
st.write(f"Training Period: {train.index[0].date()} to {train.index[-1].date()}")
st.write(f"Audit Period: {test.index[0].date()} to {test.index[-1].date()}")

st.dataframe(test.sort_index(ascending=False), use_container_width=True)
