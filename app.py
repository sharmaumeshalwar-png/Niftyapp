import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: 2-Year Liquidity Hunt Audit")

@st.cache_data(ttl=3600)
def get_audit_data():
    # 2-Year data
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    
    # Feature 1: Wick Signature (The 'Hunt' Tool)
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    # Feature 2: Liquidity Sweep Logic
    prev_low = df['Low'].shift(1).rolling(5).min().values
    prev_high = df['High'].shift(1).rolling(5).max().values
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    
    # Success definition: Trap successful if reversal occurs
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_audit_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

# 50:50 Split (Year 1 Training, Year 2 Auditing)
split = int(len(data) * 0.5)
train = data.iloc[:split]
test = data.iloc[split:]

# Train Tree on first 50%
model = DecisionTreeClassifier(max_depth=5)
model.fit(train[features], train['Success'])

# Predict Hunt Probability for every candle in the 50% Audit range
test = test.copy()
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]
test['Status'] = np.where(test['Hunt_Prob'] > 0.6, "⚠️ TRAP", "🟢 NORMAL")

st.subheader("📋 Candle-by-Candle Hunt Audit")
st.write(f"Total Candles Audited: {len(test)}")
st.dataframe(test[['Hunt_Prob', 'Status', 'Wick_Ratio', 'Is_SL_Hunt']].sort_index(ascending=False), use_container_width=True)
