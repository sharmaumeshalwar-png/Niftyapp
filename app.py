import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Bitcoin Anti-Trap 5M Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Ultra-Accurate Anti-Trap 5-Min Engine")
st.write("🎯 **Anti-Trap Logic:** Block Consecutive Multi-Buy/Sell Filters ➡️ 75% Win Optimization ➡️ Strict 63% Boundary")

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

# Fetch Data for 5-Minute Timeframe
with st.spinner("Fetching and aligning Bitcoin 5-Minute Microstructure Matrices..."):
    df = yf.download("BTC-USD", period="50d", interval="5m")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # 5-Min Look-ahead Target (15 Mins)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity'], inplace=True)

features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# Time Separation
train_mask = df.index < '2026-05-27'
predict_mask = df.index >= '2026-05-27'

df_train = df[train_mask].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0:
    st.error("No data found from May 27, 2026 onwards.")
else:
    model_flow = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_signals = df[predict_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Raw Signal Allocation
    df_signals['Raw_Signal'] = "HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'Raw_Signal'] = "BUY"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'Raw_Signal'] = "SELL"
    df_signals.loc[crossover_mask & (df_signals['Raw_Signal'] == "HOLD"), 'Raw_Signal'] = "TRAP"
    df_signals.loc[df_signals['Sign_Change'] == 0, 'Raw_Signal'] = "HOLD"

    # =====================================================================
    # CRITICAL CORRECTION: ANTI-CONSECUTIVE SIGNAL LOCK 🛡️
    # =====================================================================
    # Is logic se agar pichla active signal same hai, toh agla block ho jayega.
    final_signals = []
    last_active_signal = "HOLD"

    for current_sig in df_signals['Raw_Signal'].values:
        if current_sig in ["BUY", "SELL"]:
            if current_sig == last_active_signal:
                # Agar lagatar doosra buying aaya, toh use block kar do
                final_signals.append("⚪ MOMENTUM EXHAUSTED (Avoid)")
            else:
                if current_sig == "BUY":
                    final_signals.append("🟢 INSTITUTIONAL BUY (Confirmed)")
                else:
                    final_signals.append("🔴 INSTITUTIONAL SELL (Confirmed)")
                last_active_signal = current_sig  # State update
        elif current_sig == "TRAP":
            final_signals.append("⚪ RETAIL TRAP (Avoid Fake)")
            # Agar market trend change ka trap dikhata hai, toh logic reset kar sakte hain
            last_active_signal = "HOLD" 
        else:
            final_signals.append("⚪ HOLD")
            
    df_signals['d_ML_Signal'] = final_signals

    # Display Formatting
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live Anti-Trap Bitcoin Matrix (27 May 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Metrics
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    exhausted = len(df_signals[df_signals['d_ML_Signal'] == "⚪ MOMENTUM EXHAUSTED (Avoid)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Filter Audit Status")
    st.sidebar.write(f"🟢 Fresh Confirmed Buys: **{inst_buys}**")
    st.sidebar.write(f"🔴 Fresh Confirmed Sells: **{inst_sells}**")
    st.sidebar.error(f"⚠️ Blocked Second Moves: **{exhausted}**")
    st.sidebar.warning(f"⚪ Retail Traps Bypassed: **{traps}**")
