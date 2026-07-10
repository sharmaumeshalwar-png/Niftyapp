import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Hybrid Engine [Robust Edition]")

# =====================================================================
# DATA PIPELINE (ERROR-HANDLING ENABLED)
# =====================================================================
@st.cache_data
def load_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    # auto_adjust=True se data structure clean rehta hai
    data = yf.download("^NSEI", start=start_date, end=end_date, interval="1h", auto_adjust=True)
    return data

with st.spinner("Downloading Nifty Data..."):
    raw_df = load_data()
    
    if raw_df is None or raw_df.empty:
        st.error("Data download failed! Check your internet or Ticker.")
        st.stop()

    # Column name normalization (lowercase to avoid KeyError)
    raw_df.columns = [c.lower() for c in raw_df.columns]
    df = raw_df[['open', 'high', 'low', 'close']].ffill().dropna()

# =====================================================================
# MATHEMATICAL ENGINE & ML
# =====================================================================
# Kalman Filter logic
def apply_kalman(data, initial_p=100.0, q=0.0001, r=2.5):
    x, p = data[0], initial_p
    res = []
    for z in data:
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p *= (1 - k)
        res.append(x)
    return res

df['a_close'] = df['close']
df['b_kalman'] = apply_kalman(df['a_close'].values)
df['c_combined'] = df['a_close'] - df['b_kalman']

# Features
df['order_imb'] = (df['a_close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
df['norm_gap'] = df['c_combined'] / (df['c_combined'].rolling(24).std() + 1e-10)

features = ['c_combined', 'order_imb', 'norm_gap']
df['target'] = np.where(df['a_close'] > df['a_close'].shift(1), 1, 0)
df_clean = df.dropna()

# ML Engine
split_idx = int(len(df_clean) * 0.5)
train, predict = df_clean.iloc[:split_idx], df_clean.iloc[split_idx:].copy()

model = RandomForestClassifier(n_estimators=100, max_depth=3).fit(train[features], train['target'])
predict['prob_up'] = model.predict_proba(predict[features])[:, 1]

# Display
st.subheader("📋 Last 50 Signals")
predict['signal'] = np.where(predict['prob_up'] > 0.55, "🟢 BUY", "⚪ NEUTRAL")
st.dataframe(predict[['close', 'prob_up', 'signal']].tail(50))
