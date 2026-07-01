import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty July 2025 Engine", layout="wide")
st.title("🦅 Nifty 50 Strict Microstructure Engine (July 2025 - Present)")
st.write("🎯 **Timeline Logic:** Displaying Full Refined Matrix Data strictly starting from **1 July 2025** up to Today")

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

# Fetch Data automatically from free rolling source
with st.spinner("Aligning continuous database from July 2025 onwards..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    # Fetching from early 2024 to give the model massive pre-training context before 1 July 2025
    start_date = (datetime.now() - timedelta(days=710)).strftime('%Y-%m-%d')
    
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("Data source timeout. Please click Reboot App on the dashboard.")
        st.stop()

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # REFINED MICROSTRUCTURE FEATURES
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Setup (3 Hours Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity', 'Target'], inplace=True)

# Extended Feature Matrix 
features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# STRICT SEPARATION FOR VISUAL WINDOW (Pre-Train before July 2025, Predict from July 2025)
train_mask = (df.index < '2025-07-01')
display_mask = (df.index >= '2025-07-01')

X_train = df.loc[train_mask, features_matrix]
y_train = df.loc[train_mask, 'Target']
X_display = df.loc[display_mask, features_matrix]

if len(X_display) == 0:
    st.error("Timeline shift failed. No data found after 1 July 2025.")
else:
    # Stable Forest Model
    model_flow = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
    
    # Secure training boundary protection
    if len(X_train) > 20:
        model_flow.fit(X_train, y_train)
    else:
        model_flow.fit(df[features_matrix], df['Target'])

    probabilities = model_flow.predict_proba(X_display)
    df_signals = df[display_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize Signals Block
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # STRICT 63% ACCURACY FILTER RESTORED
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
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
    st.subheader(f"📋 Nifty 50 Execution Matrix (1 July 2025 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Microstructure Cumulative Stats")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Fake Traps Blocked: **{traps}**")
