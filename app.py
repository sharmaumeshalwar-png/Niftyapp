import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Probability & Consensus Engine [Fixed]")

# EMA list ko global rakha taaki har jagah access ho sake
EMAS = [20, 50, 100, 200]

@st.cache_data(ttl=3600)
def get_consensus_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    # Har EMA ke liye directional signal
    for e in EMAS:
        ema_val = df['Price'].ewm(span=e).mean()
        df[f'Signal_{e}'] = (ema_val.diff() > 0).astype(int)
        
    df['Consensus'] = df[[f'Signal_{e}' for e in EMAS]].sum(axis=1)
    df['Move'] = df['Price'].shift(-1) - df['Price']
    df['Target_Up'] = (df['Move'] > 0).astype(int)
    
    return df.dropna()

data = get_consensus_data()

# Training Logic
X = data[[f'Signal_{e}' for e in EMAS]]
y = data['Target_Up']

split = int(len(data) * 0.5)
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X.iloc[:split], y.iloc[:split])

# Prediction
data['Prob_Up'] = model.predict(X)
data['Action'] = data['Prob_Up'].apply(lambda x: "BUY" if x > 0.6 else ("SELL" if x < 0.4 else "HOLD"))

st.subheader("📋 Consensus Audit (EMA Hierarchy)")
st.dataframe(data.sort_index(ascending=False).head(20), use_container_width=True)
