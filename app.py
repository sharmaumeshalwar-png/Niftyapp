import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Free ML Bot", layout="wide")
st.title("🛡️ Nifty 50 Free Tokenless ML Engine")
st.write("Target Core: Trigger on 'c' Sign Flip ➡️ Analyze Simulated Microstructure Imbalance (No Token Required)")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Your exact strict constraint
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

# Fetch Data automatically from free source
with st.spinner("Connecting to Free Data Engine and tracking microstructure..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman Filter (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # c = Combined Matrix Gap
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # ADVANCED MICROSTRUCTURE PROXY (Decodes institutional order flow fields)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Setup (3 Hours Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Flow_Velocity', 'Target'], inplace=True)

# Clean Feature Matrix
features_matrix = ['c_Combined', 'Order_Imbalance', 'Flow_Velocity']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ONLY on exact crossover hours
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Historical dataset compression failed. Please restart.")
else:
    # Model Setup
    model_d = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize Signals Block
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Stable 63% threshold for confirmation
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
    
    # Filter out retail trap noise
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
    
    # Absolute hold lock when no sign flip happens
    df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

    # Clean display frame extraction
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formatting outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    # Main Grid Data Presentation
    st.subheader(f"📋 Nifty 50 Free Execution Matrix (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Microstructure Stats")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Fake Traps Blocked: **{traps}**")
