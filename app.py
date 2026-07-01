import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Active Engine", layout="wide")
st.title("🦅 Nifty 50 Aggressive Sequenced ML Engine (2025 - Present)")
st.write("🎯 **Fixed Logic:** Threshold optimized to 55% with deeper trees to unlock trade frequency from 1 Jan 2025")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter 0.001)
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

# Fetch Data with Dynamic Safety Limit
with st.spinner("Unlocking active signal flow matrices..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=700)).strftime('%Y-%m-%d')
    
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("Yahoo Finance timeout. Please retry via Reboot App.")
        st.stop()

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Strict Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Advanced Microstructure Proxy (Amplified Cues)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Strict Target Definition (3 Hours Forward)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Flow_Velocity', 'Target'], inplace=True)

# Feature Matrix
features_matrix = ['c_Combined', 'Order_Imbalance', 'Flow_Velocity']

# SEQUENTIAL BLOCK TRAINING 
train_base_mask = (df.index < '2025-01-01')
display_mask = (df.index >= '2025-01-01')

X_train_base = df.loc[train_base_mask, features_matrix]
y_train_base = df.loc[train_base_mask, 'Target']
X_display = df.loc[display_mask, features_matrix]

if len(X_display) == 0:
    display_mask = (df.index >= df.index[0] + timedelta(days=30))
    X_display = df.loc[display_mask, features_matrix]

# Deeper trees to capture trade sensitivity
master_model = RandomForestClassifier(n_estimators=350, max_depth=8, min_samples_split=4, random_state=42)

if len(X_train_base) > 10:
    master_model.fit(X_train_base, y_train_base)
else:
    master_model.fit(df[features_matrix], df['Target'])

probabilities = master_model.predict_proba(X_display)
df_signals = df[display_mask].copy()
df_signals['Prob_Down'] = probabilities[:, 0]
df_signals['Prob_Up'] = probabilities[:, 1]

# Default Layout State
df_signals['d_ML_Signal'] = "⚪ HOLD"
crossover_mask = df_signals['Sign_Change'] == 1

# Optimized 55% threshold for active and frequent signal updates
df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.55), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.55), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

# Clean display frame extraction
clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
display_df = df_signals[clean_display_cols].copy()

# Formatting numbers
display_df['a_Close'] = display_df['a_Close'].round(2)
display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
display_df['c_Combined'] = display_df['c_Combined'].round(4)
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

# Main Grid Rendering
st.subheader(f"📋 Active Nifty 50 Timeline (1 Jan 2025 - Present)")
st.dataframe(display_df, use_container_width=True, height=800)

# Sidebar Filter Counter Metrics
total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

st.sidebar.header("📊 Filter Audit (2025-2026)")
st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
st.sidebar.write(f"🟢 Active Validated Buys: **{inst_buys}**")
st.sidebar.write(f"🔴 Active Validated Sells: **{inst_sells}**")
st.sidebar.warning(f"⚪ Noisy Traps Filtered: **{traps}**")
