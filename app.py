import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty 50 2-Year Engine", layout="wide")
st.title("📊 Nifty 50 [2-Year Historical] Hybrid Engine")

# =====================================================================
# MATHEMATICAL ENGINE
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    x = data_array[0]
    p = initial_p
    q, r = 0.0001, 2.5
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# DATA FETCHING & CLEANING
# =====================================================================
@st.cache_data
def get_nifty_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # Downloading Daily data for 2 years
    raw_df = yf.download("^NSEI", start=start_date, end=end_date, interval="1d")
    
    # FIX: Handling MultiIndex columns
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    df = raw_df.reset_index()
    df.dropna(inplace=True)
    return df

with st.spinner("Fetching and Calibrating 2-Year Nifty Matrices..."):
    df = get_nifty_data()
    
    # Ensure columns exist and are numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col])

    # Base Matrix Calculations
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(1), 1, 0)
    
    # Final Clean
    df_clean = df.dropna(subset=features_matrix + ['Target']).copy()

# =====================================================================
# DYNAMIC 50:50 SPLIT ENGINE
# =====================================================================
split_idx = int(len(df_clean) * 0.50)
df_train, df_predict = df_clean.iloc[:split_idx], df_clean.iloc[split_idx:]

X_train, y_train = df_train[features_matrix], df_train['Target']
model_flow = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

# Predictions
probs = model_flow.predict_proba(df_predict[features_matrix])
df_predict['Prob_Up'] = probs[:, 1]
df_predict['Prob_Down'] = probs[:, 0]

# Logic Loop
accumulator = 0
signals = []
for i in range(len(df_predict)):
    if df_predict['Prob_Up'].iloc[i] >= 0.55: accumulator += 1
    elif df_predict['Prob_Down'].iloc[i] >= 0.55: accumulator -= 1
    accumulator = max(-5, min(5, accumulator))
    
    if accumulator == 5: sig = "🟢 STRONG BUY"
    elif accumulator == -5: sig = "🔴 STRONG SELL"
    else: sig = f"🔄 Neutral (Score: {accumulator})"
    signals.append(sig)

df_predict['d_ML_Signal'] = signals

# Display
st.subheader("📋 2-Year Nifty Dual Momentum Results (50:50 Split)")
st.dataframe(df_predict[['Date', 'a_Close', 'Prob_Up', 'd_ML_Signal']].tail(50))

st.write("---")
st.write("✅ **Engine Status:** 50% Train / 50% Predict | Data: 2 Years Daily | ML: Random Forest")
