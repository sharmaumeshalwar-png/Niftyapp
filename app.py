import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🚀 Nifty 50: Liquidity Hunt Tree Decoder [No-Crash Mode]")

@st.cache_data(ttl=3600)
def get_hunt_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    
    # 1. Geometry: Wick vs Body (Is it a sweep?)
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    # 2. Hunting Logic: Price broke previous 5-hour low/high
    df['Prev_Low'] = df['Low'].shift(1).rolling(5).min()
    df['Prev_High'] = df['High'].shift(1).rolling(5).max()
    df['Is_SL_Hunt'] = ((df['Low'] < df['Prev_Low']) | (df['High'] > df['Prev_High'])).astype(int)
    
    # 3. Success: Kya hunt ke baad price reversal hua?
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_hunt_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

# Tree Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

# Probability Audit (Table Format)
test = test.copy()
test['Hunt_Probability'] = model.predict_proba(test[features])[:, 1]
test['Decision'] = np.where(test['Hunt_Probability'] > 0.6, "TRAP CONFIRMED", "NORMAL")

st.subheader("📋 Tree Decision Audit (2-Year Backtest)")
st.dataframe(test[['Hunt_Probability', 'Decision', 'Is_SL_Hunt', 'Wick_Ratio']].sort_index(ascending=False).head(20), use_container_width=True)
