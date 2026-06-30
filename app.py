import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Pure Pattern Engine", layout="wide")
st.title("🦅 Nifty 50 Pure Sign-History ML Bot")
st.write("🎯 **Core Logic:** No Indicators | Analyze 'c' Sign Change History ➡️ Detect False Flips vs Real Trends")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Your exact 0.001 constraint
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# Fetch Data directly for NIFTY 50 Index
with st.spinner("Reading Nifty historical patterns..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    # Using Nifty 50 Index instead of BeES
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
    
    # HISTORY TRACKING (ML reads previous states of 'c' to learn why it flipped or trended)
    df['c_Lag1'] = df['c_Combined'].shift(1)
    df['c_Lag2'] = df['c_Combined'].shift(2)
    df['Price_Velocity'] = df['a_Close'].diff(1)
    
    # Target definition (Forward Trend Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['c_Lag2', 'Price_Velocity', 'Target'], inplace=True)

# Pure History Features for ML (No external indicators like VWAP/Volume/MACD)
features_matrix = ['c_Combined', 'c_Lag1', 'c_Lag2', 'Price_Velocity']

# Train/Test Windows
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train only on sign change moments from history
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']

X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Historical computation mismatch. Please restart the app.")
else:
    # Model configuration to decode trend physics
    model_d = RandomForestClassifier(n_estimators=250, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Signal Setup
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    
    # Rule: Trigger only at crossover nodes
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Balanced 60% surety threshold on pure price pattern
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.60), 'd_ML_Signal'] = "🟢 TREND UP (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.60), 'd_ML_Signal'] = "🔴 TREND DOWN (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] < 0.60) & (df_signals['Prob_Down'] < 0.60), 'd_ML_Signal'] = "⚪ FALSE FLIP (Avoid)"

    # Clean display configuration
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Data View
    st.subheader(f"📋 Nifty 50 Structure Execution Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Verification Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    real_trends_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 TREND UP (Confirmed)"])
    real_trends_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 TREND DOWN (Confirmed)"])
    false_flips = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FALSE FLIP (Avoid)"])

    st.sidebar.header("📊 Nifty History Statistics")
    st.sidebar.write(f"Total Sign Flips: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Trends Up: **{real_trends_up}**")
    st.sidebar.write(f"🔴 Confirmed Trends Down: **{real_trends_dn}**")
    st.sidebar.write(f"⚪ False/Choppy Flips Filtered: **{false_flips}**")
