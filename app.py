import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50 Probability & Consensus Engine")

@st.cache_data(ttl=3600)
def get_consensus_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False)
    df = df[['Close']].ffill()
    df.columns = ['Price']
    
    emas = [20, 50, 100, 200]
    # Har EMA ke liye ek "Directional Signal" (Slope)
    for e in emas:
        ema_val = df['Price'].ewm(span=e).mean()
        # 1 = Up (Slope positive), 0 = Down (Slope negative)
        df[f'Signal_{e}'] = (ema_val.diff() > 0).astype(int)
        
    # Consensus: Agar 4-ro up hai toh 4, sab down hai toh 0
    df['Consensus'] = df[[f'Signal_{e}' for e in emas]].sum(axis=1)
    
    # Target move
    df['Move'] = df['Price'].shift(-1) - df['Price']
    df['Target_Up'] = (df['Move'] > 0).astype(int)
    
    return df.dropna()

data = get_consensus_data()

# Model: Probability estimate
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
X = data[[f'Signal_{e}' for e in emas]]
y = data['Target_Up']

# Train on 50%
split = int(len(data) * 0.5)
model.fit(X.iloc[:split], y.iloc[:split])

# Probability predict karo
data['Prob_Up'] = model.predict(X)
data['Action'] = data['Prob_Up'].apply(lambda x: "BUY" if x > 0.6 else ("SELL" if x < 0.4 else "HOLD"))

st.subheader("📋 Consensus Audit (EMA Hierarchy)")
st.dataframe(data.sort_index(ascending=False).head(20), use_container_width=True)
