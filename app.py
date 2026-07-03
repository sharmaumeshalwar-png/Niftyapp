import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Nifty Institutional 1H Engine", layout="wide")
st.title("🦅 Nifty 50 Ultra-Accurate Institutional 1-Hour Engine")
st.write("🎯 **Max Training Mode:** Learning from past 2 years ➡️ Signals active from 1 Jan 2026 ➡️ 63% Strict Filter")

# ==========================================
# MATHEMATICAL ENGINE (Kalman Filter 0.001)
# ==========================================
def apply_kalman_filter_strict(price_array):
    if len(price_array) == 0:
        return []
    x = price_array[0]
    p = 50.0
    q = 0.001
    r = 0.1
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# Fetch Data for 1-Hour Timeframe 
# 'period="720d"' se pichle ~2 saal ka saara data load hoga taaki model max seekh sake
with st.spinner("Fetching max historical 1-Hour data for deep learning..."):
    df = yf.download("^NSEI", period="720d", interval="1h")

# Bug Fix: Multi-Index Column Flattening
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if len(df) == 0:
    st.error("YFinance API Timeout or No Data found. Please refresh the dashboard.")
    st.stop()

# Index handling to ensure datetime string comparison works perfectly
df.index = pd.to_datetime(df.index)

# Base Matrix Definition
df['a_Close'] = df['Close']
df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
df['c_Combined'] = df['a_Close'] - df['b_Kalman']

# Sign Flip Lock
df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
df['Sign_Change'] = df['Sign_Change'].astype(int)

# ==========================================
# MICROSTRUCTURE FEATURES (Optimized for 1-Hour Frame)
# ==========================================
df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
df['Normalized_Gap'] = df['c_Combined'] / rolling_std

df['Flow_Velocity'] = df['c_Combined'].diff(1)

# 1-Hour ke hisab se Look-ahead Target (3 candles forward = 3 Hours lookahead)
df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)

# Clear Feature NaNs (Live Rows are kept safe here!)
df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity'], inplace=True)

# Feature list mapping
features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# ==========================================
# STRICT SEPARATION OF TIME LOGIC (Signals from 1 Jan 2026)
# ==========================================
# 1 Jan 2026 se pehle ka pichla saara data (approx 2 saal) training me jayega
train_mask = df.index < '2026-01-01'
# 1 Jan 2026 se lekar aaj tak ka data prediction/hints me dikhega
predict_mask = df.index >= '2026-01-01'

# Training Set
df_train = df[train_mask].dropna(subset=['Target'])

X_train = df_train[features_matrix]
y_train = df_train['Target']

# Prediction Set
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0:
    st.error("No data found from Jan 1, 2026 onwards for prediction.")
else:
    # Model Setup
    model_flow = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
    model_flow.fit(X_train, y_train)
    
    # ML Inference Processing
    probabilities = model_flow.predict_proba(X_predict)
    df_signals = df[predict_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]
    
    # Signal Logic
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # STRICT 63% ACCURACY FILTER
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
    df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

    # Display Engine Extraction
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()
    
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    
    # Top Row = Absolute Live Running Candle Hint
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')
    
    # Main Grid Presentation
    st.subheader("📋 Live 1-Hour Nifty 50 Execution Matrix (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)
    
    # Sidebar Metric Counter Auditor
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])
    
    st.sidebar.header("📊 Production Audit (Post Jan 01)")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Fake Traps Filtered: **{traps}**")
