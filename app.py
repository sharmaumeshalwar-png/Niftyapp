import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: Exact Strike Signal Engine [Ultra-Stable]")

@st.cache_data(ttl=3600)
def get_signal_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    df['Close_Price'] = df['Close']
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    prev_low = df['Low'].shift(1).rolling(5).min().bfill().values
    prev_high = df['High'].shift(1).rolling(5).max().bfill().values
    
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_signal_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

# Prediction
test = test.copy()
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]

# STABLE LOGIC: Use direct series access instead of .loc dataframe indexing
test['Call'] = "WAITING"
mask = test['Hunt_Prob'] > 0.6
prev_close = test['Close'].shift(1).bfill()

# Update Call column
down_mask = mask & (test['Close'] > prev_close)
up_mask = mask & (test['Close'] <= prev_close)

# Direct Series assignment
test.loc[down_mask, 'Call'] = test['Close'].apply(lambda x: f"{x:.0f} se DOWN aayega")
test.loc[up_mask, 'Call'] = test['Close'].apply(lambda x: f"{x:.0f} se UP jayega")

st.subheader("📋 2-Year Audit: Exact Strike Calls")
st.dataframe(test[['Close_Price', 'Hunt_Prob', 'Call']].sort_index(ascending=False).head(20), use_container_width=True)
