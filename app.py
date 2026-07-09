import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: Exact Strike Signal Engine [Fixed]")

@st.cache_data(ttl=3600)
def get_signal_data():
    # 2 Year Data
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    
    # Feature Engineering
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    # Hunting logic (5-hour window) - Fixed bfill()
    prev_low = df['Low'].shift(1).rolling(5).min().bfill().values
    prev_high = df['High'].shift(1).rolling(5).max().bfill().values
    
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    
    # Success Logic
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_signal_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

# 50:50 Audit split
split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

test = test.copy()
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]

# Strike Signal Engine
test['Call'] = "WAITING"
mask = test['Hunt_Prob'] > 0.6
prev_close = test['Close'].shift(1).bfill()

# Logic for Strike Price
down_cond = mask & (test['Close'] > prev_close)
up_cond = mask & (test['Close'] <= prev_close)

# Vectorized string formatting to avoid TypeError/Arrow issues
test.loc[down_cond, 'Call'] = test.loc[down_cond, 'Close'].map(lambda x: f"{x:.0f} se DOWN aayega")
test.loc[up_cond, 'Call'] = test.loc[up_cond, 'Close'].map(lambda x: f"{x:.0f} se UP jayega")

st.subheader("📋 2-Year Audit: Exact Strike Calls")
st.dataframe(test[['Close', 'Hunt_Prob', 'Call']].sort_index(ascending=False).head(20), use_container_width=True)
