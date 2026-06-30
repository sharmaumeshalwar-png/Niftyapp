import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Fixed Engine", layout="wide")
st.title("🦅 Nifty 50 Strict Sign-VIX Matcher")
st.write("🎯 **Core Logic:** Focus ONLY on 'c' Sign Change ➡️ Match with VIX Level ➡️ Predict True Trend vs False Flip")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Direct strict parameter
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# Data Engine Pipeline
with st.spinner("Aligning Nifty and VIX historical crossover points..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    df_nifty = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    df_vix = yf.download("^INDIAVIX", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df_nifty.columns, pd.MultiIndex): df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)

    # Clean Sync
    df = df_nifty[['High', 'Low', 'Close']].copy()
    df['a_Close'] = df['Close']
    df['VIX_Close'] = df_vix['Close']
    df.ffill(inplace=True)
    
    # Base Math
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # HARD LOCK: Track strict sign flip points
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Simple Velocity for ML context
    df['Price_Velocity'] = df['a_Close'].diff(1)
    
    # Target (3 Hours Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VIX_Close', 'Price_Velocity', 'Target'], inplace=True)

# Clean, Un-fuzzed Feature Matrix
features_matrix = ['c_Combined', 'VIX_Close', 'Price_Velocity']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# CRITICAL FILTER: Train ML ONLY on the exact moments when sign flipped
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("No historical crossovers found. Please restart the app.")
else:
    # Stable classifier setup
    model_d = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Default State
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    
    # Condition: Only apply signals at crossover points
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Practical 62% threshold for stable match
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.62), 'd_ML_Signal'] = "🟢 TREND UP (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.62), 'd_ML_Signal'] = "🔴 TREND DOWN (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ FALSE FLIP (Avoid)"
    
    # If no sign change happened, force absolute HOLD
    df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

    # Display Framing
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Nifty 50 Strict Crossover Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    confirmed_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 TREND UP (Confirmed)"])
    confirmed_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 TREND DOWN (Confirmed)"])
    avoided = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FALSE FLIP (Avoid)"])

    st.sidebar.header("📊 Filter Statistics")
    st.sidebar.write(f"Total Flips Detected: **{total_flips}**")
    st.sidebar.write(f"🟢 Real Trends Up: **{confirmed_up}**")
    st.sidebar.write(f"🔴 Real Trends Down: **{confirmed_dn}**")
    st.sidebar.write(f"⚪ False Flips Blocked: **{avoided}**")
