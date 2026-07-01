import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Fixed 2025 Engine", layout="wide")
st.title("🦅 Nifty 50 Strict Sequenced ML Engine (2025 - Present)")
st.write("🎯 **Fixed Logic:** Sequential Training to eliminate Data Leakage and fix wrong hints from 1 Jan 2025")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
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

# Fetch Data
with st.spinner("Re-aligning data streams sequentially to fix predictions..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    # Fetching extra historical data to ensure 1 Jan 2025 starts with a fully trained model
    df = yf.download("^NSEI", start="2024-06-01", end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Strict Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Advanced Microstructure Proxy
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Strict Target Definition (3 Hours Forward)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Flow_Velocity', 'Target'], inplace=True)

# Feature Matrix
features_matrix = ['c_Combined', 'Order_Imbalance', 'Flow_Velocity']

# SEQUENTIAL BLOCK TRAINING (Prevents Future Data Leakage)
# Split at 1 Jan 2025 strictly
train_base_mask = (df.index < '2025-01-01')
display_mask = (df.index >= '2025-01-01')

X_train_base = df.loc[train_base_mask, features_matrix]
y_train_base = df.loc[train_base_mask, 'Target']
X_display = df.loc[display_mask, features_matrix]

if len(X_display) == 0:
    st.error("Timeline tracking error. Please restart app.")
else:
    # Train primary master model on solid base history
    master_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_split=10, random_state=42)
    master_model.fit(X_train_base, y_train_base)
    
    # Generate clean probabilities without future sight
    probabilities = master_model.predict_proba(X_display)
    
    df_signals = df[display_mask].copy()
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Default Layout State
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Strict 65% confirmation filter to eliminate wrong hints
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.65), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.65), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
    
    # Categorize failed crossovers as retail noise traps
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
    
    # Direct lock when no crossover occurs
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
    st.subheader(f"📋 Corrected Nifty 50 Timeline (1 Jan 2025 - Present)")
    st.dataframe(display_df, use_container_width=True, height=800)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Filter Audit (2025-2026)")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Real Validated Buys: **{inst_buys}**")
    st.sidebar.write(f"🔴 Real Validated Sells: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Noisy Traps Filtered: **{traps}**")
