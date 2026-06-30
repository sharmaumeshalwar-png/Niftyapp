import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Velocity ML Bot", layout="wide")
st.title("⚡ Nifty 50 Pure Velocity ML Engine")
st.write("🎯 **Core Logic:** Trigger on 'c' Sign Change ➡️ Analyze Price Velocity & Acceleration ➡️ Catch True Trends")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Custom strict constraint
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
with st.spinner("Decoding price velocity physics..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman Filter (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # c = Combined Data Matrix (Gap between raw and smooth price)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Track when 'c' flips sign
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # PURE VELOCITY & ACCELERATION (No indicators, just pure price speed)
    df['Price_Velocity'] = df['a_Close'].diff(1)  # Speed of current candle
    df['Velocity_Lag1'] = df['Price_Velocity'].shift(1)  # Speed of previous candle
    df['Acceleration'] = df['Price_Velocity'].diff(1)  # Change in speed
    
    # Target definition (Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Price_Velocity', 'Velocity_Lag1', 'Acceleration', 'Target'], inplace=True)

# Features Matrix focused ONLY on Velocity metrics
features_matrix = ['Price_Velocity', 'Velocity_Lag1', 'Acceleration', 'c_Combined']

# Train (2025) and Test (2026 - Today) Masks
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ML ONLY on velocity behaviors during crossover nodes
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']

X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Velocity data alignment issue. Please reboot the app.")
else:
    # Random Forest trained strictly to analyze speed dynamics
    model_d = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize clean execution signal
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    
    # Condition: ONLY trigger on sign flips
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Strict 65% probability filter on velocity
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.65), 'd_ML_Signal'] = "🟢 TREND UP (Velocity Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.65), 'd_ML_Signal'] = "🔴 TREND DOWN (Velocity Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] < 0.65) & (df_signals['Prob_Down'] < 0.65), 'd_ML_Signal'] = "⚪ FALSE FLIP (Avoid)"

    # Clean display subset as requested
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Web UI Layout Render
    st.subheader(f"📋 Nifty 50 Pure Velocity Execution Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    real_trends_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 TREND UP (Velocity Confirmed)"])
    real_trends_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 TREND DOWN (Velocity Confirmed)"])
    false_flips = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FALSE FLIP (Avoid)"])

    st.sidebar.header("📊 Velocity Engine Stats")
    st.sidebar.write(f"Total Sign Changes: **{total_flips}**")
    st.sidebar.write(f"🟢 High-Speed Buy: **{real_trends_up}**")
    st.sidebar.write(f"🔴 High-Speed Sell: **{real_trends_dn}**")
    st.sidebar.write(f"⚪ Low-Speed Fakes: **{false_flips}**")
