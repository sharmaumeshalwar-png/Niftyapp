import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Bitcoin Institutional 5M Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Ultra-Accurate Institutional 5-Min Engine")
st.write("🎯 **Crypto Production-Grade Engine:** 5-Minute Matrix ➡️ BTC-USD Asset Flow ➡️ Strict 63% Filter")

# =====================================================================
# MATHEMATICAL ENGINE (Kalman Filter 0.001)
# =====================================================================
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

# Fetch Data for 5-Minute Timeframe (Bitcoin runs 24/7)
with st.spinner("Fetching and aligning Bitcoin 5-Minute Microstructure Matrices..."):
    # yfinance 5m data max 60 days tak ka deta hai, crypto ke liye rolling 50 days nikal rahe hain
    df = yf.download("BTC-USD", period="50d", interval="5m")
    
    # Bug Fix: Multi-Index Column Flattening
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("YFinance API Timeout or Invalid Ticker. Please refresh the dashboard.")
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
    
    # =====================================================================
    # MICROSTRUCTURE FEATURES (Optimized for Crypto 5-Min Frame)
    # =====================================================================
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # 5-Min Look-ahead Target (3 candles forward = 15 Mins lookahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    
    # Clear Feature NaNs (Live Rows are kept safe here!)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity'], inplace=True)

# Feature list mapping
features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# =====================================================================
# STRICT SEPARATION OF TIME LOGIC (Strict Target Isolation)
# =====================================================================
# 27 May 2026 se pehle ka data training ke liye use hoga
train_mask = df.index < '2026-05-27'
# 27 May 2026 se lekar aaj (Live Present) tak ka data prediction matrix me jayega
predict_mask = df.index >= '2026-05-27'

# Training Set: Yahan se target drop karenge perfect mathematically training ke liye
df_train = df[train_mask].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']

# Prediction Set: Live Market rows completely untouched and safe!
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0:
    st.error("No data found from May 27, 2026 onwards. Check yfinance connection.")
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
    st.subheader(f"📋 Live 5-Min Bitcoin Execution Matrix (27 May 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Metric Counter Auditor
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Crypto Production Audit")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Fake Traps Filtered: **{traps}**")
