import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Hidden Pattern Engine", layout="wide")
st.title("👁️ Nifty 50 Hidden-Pattern ML Engine")
st.write("Target Engine: Deep Pattern Recognition on 'c' Sign Flip Nodes using Physics Analytics")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Your strict constraint
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# Fetch Data for NIFTY 50 Index
with st.spinner("ML is decoding hidden historical matrix patterns..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman Filter (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # c = Combined Data Matrix (Price vs Filter gap)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Track when 'c' flips sign
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # PURE PHYSICS FEATURES (Things only ML can corelate during a flip)
    df['Price_Velocity'] = df['a_Close'].diff(1)
    df['Velocity_Lag1'] = df['Price_Velocity'].shift(1)
    
    # Hidden Feature 1: Volatility Squish (Did the candle compress before flip?)
    df['Volatility_Squish'] = df['High'] - df['Low']
    
    # Hidden Feature 2: Velocity Elasticity (Rolling kinetic build-up of past 3 hours)
    df['Elasticity'] = df['Price_Velocity'].rolling(window=3).sum()
    
    # Target definition (Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Price_Velocity', 'Volatility_Squish', 'Elasticity', 'Target'], inplace=True)

# Features Matrix built with the hidden variables discovered by ML logic
features_matrix = ['Price_Velocity', 'Velocity_Lag1', 'Volatility_Squish', 'Elasticity', 'c_Combined']

# Train (2025) and Test (2026 - Today) Masks
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ML ONLY on cross-nodes to extract hidden traits
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Matrix generation mismatch. Please reboot application.")
else:
    # Random forest given deep trees to find complex pattern joints
    model_d = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize execution block
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # 65% surety filter operating on hidden dimensions
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.65), 'd_ML_Signal'] = "🟢 TREND UP (Pattern Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.65), 'd_ML_Signal'] = "🔴 TREND DOWN (Pattern Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] < 0.65) & (df_signals['Prob_Down'] < 0.65), 'd_ML_Signal'] = "⚪ FALSE FLIP (Avoid)"

    # Clean display subset
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Web UI Layout Render
    st.subheader(f"📋 Nifty 50 Hidden Mechanics Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    real_trends_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 TREND UP (Pattern Confirmed)"])
    real_trends_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 TREND DOWN (Pattern Confirmed)"])
    false_flips = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FALSE FLIP (Avoid)"])

    st.sidebar.header("📊 Intelligence Analytics")
    st.sidebar.write(f"Total Sign Changes: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Trends Up: **{real_trends_up}**")
    st.sidebar.write(f"🔴 Confirmed Trends Down: **{real_trends_dn}**")
    st.sidebar.write(f"⚪ Shaky Fakes Deflected: **{false_flips}**")
