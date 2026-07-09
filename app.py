import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: Exact Action Signal Engine")

@st.cache_data(ttl=3600)
def get_signal_data():
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    
    # 1. Price Data
    df['Close_Price'] = df['Close']
    
    # 2. Geometry: Wick vs Body
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    # 3. Hunting Logic
    prev_low = df['Low'].shift(1).rolling(5).min().values
    prev_high = df['High'].shift(1).rolling(5).max().values
    df['Is_SL_Hunt'] = ((df['Low'].values < prev_low) | (df['High'].values > prev_high)).astype(int)
    
    # 4. Target Label (Success = Trap Reversal)
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_signal_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split], data.iloc[split:]

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

# Prediction with Signal Logic
test = test.copy()
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]

# Exact Directional Logic
# Agar Prob > 60% hai, toh Trap hai -> Opposite Direction ka signal do
test['Signal'] = np.where(test['Hunt_Prob'] > 0.6, 
                          np.where(test['Close'] > test['Close'].shift(1), "DOWN EXPECTED", "UP EXPECTED"), 
                          "SIDEWAYS/NORMAL")

st.subheader("📋 Exact Directional Signal Audit")
st.dataframe(test[['Close_Price', 'Hunt_Prob', 'Signal']].sort_index(ascending=False).head(20), use_container_width=True)
