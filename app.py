import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")
st.title("🎯 Nifty 50: Exact Strike Signal Engine [Absolute Fix]")

@st.cache_data(ttl=3600)
def get_signal_data():
    # Load and clean data (ensure Series format)
    df = yf.download("^NSEI", period="2y", interval="1h", progress=False).ffill()
    # If columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df['Close_Price'] = df['Close']
    df['Wick_Ratio'] = (df['High'] - df['Low']) / (abs(df['Close'] - df['Open']) + 0.001)
    
    prev_low = df['Low'].shift(1).rolling(5).min().bfill()
    prev_high = df['High'].shift(1).rolling(5).max().bfill()
    
    df['Is_SL_Hunt'] = ((df['Low'] < prev_low) | (df['High'] > prev_high)).astype(int)
    df['Success'] = ((df['Close'].shift(-2) - df['Close']) * np.sign(df['Close'] - df['Open']) < 0).astype(int)
    
    return df.dropna()

data = get_signal_data()
features = ['Wick_Ratio', 'Is_SL_Hunt']

split = int(len(data) * 0.5)
train, test = data.iloc[:split].copy(), data.iloc[split:].copy()

model = DecisionTreeClassifier(max_depth=4)
model.fit(train[features], train['Success'])

# Prediction
test['Hunt_Prob'] = model.predict_proba(test[features])[:, 1]
prev_close = test['Close'].shift(1).bfill()

# Create 'Call' column safely
test['Call'] = "WAITING"
mask = test['Hunt_Prob'] > 0.6
price_str = test['Close'].astype(int).astype(str)

# Apply conditions using simple Series boolean indexing
test.loc[mask & (test['Close'] > prev_close), 'Call'] = price_str + " se DOWN aayega"
test.loc[mask & (test['Close'] <= prev_close), 'Call'] = price_str + " se UP jayega"

st.subheader("📋 2-Year Audit: Exact Strike Calls")
st.dataframe(test[['Close_Price', 'Hunt_Prob', 'Call']].sort_index(ascending=False).head(20), use_container_width=True)
