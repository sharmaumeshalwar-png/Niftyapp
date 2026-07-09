import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: Exact Action Signal Engine [Stable]")

@st.cache_data(ttl=3600)
def get_signal_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    df['Close_Price'] = df['Close']
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    prev_low = df['Low'].shift(1).rolling(5).min().values
    prev_high = df['High'].shift(1).rolling(5).max().values
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    return df.dropna()

data = get_signal_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

test = test.copy()
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]

# FIX: String data handling to prevent Arrow error
test['Signal'] = "SIDEWAYS/NORMAL"
mask = test['Hunt_Prob'] > 0.6
test.loc[mask, 'Signal'] = np.where(test.loc[mask, 'Close'] > test.loc[mask, 'Close'].shift(1), "DOWN EXPECTED", "UP EXPECTED")

st.subheader("📋 Exact Directional Signal Audit")
st.dataframe(test[['Close_Price', 'Hunt_Prob', 'Signal']].sort_index(ascending=False).head(20), use_container_width=True)
