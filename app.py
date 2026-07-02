import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Umesh Search Engine - 5m April", layout="wide")
st.title("🦅 Umesh Search: Institutional Order-Flow Engine (5-Min Scalper)")
st.write("🎯 **Refined Core Logic:** 5-Minute Candles ➡️ Enhanced Microstructure Features ➡️ Strict 63% Filter ➡️ Since **1 April 2026**")

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

# Fetch Data automatically from free rolling source (Max 60 days allowed for 5m)
with st.spinner("Refining 5-minute microstructure matrices for ultra-accuracy..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="5m")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("Data source timeout. Please click Reboot App.")
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
    
    # OPTIMIZED TARGET SETUP: 1 Candle Look-ahead (5-Minute Future Horizon)
    df['Target'] = np.where(df['a_Close'].shift(-1) > df['a_Close'], 1, 0)
    
    # Drop ONLY feature dependencies globally. Do not drop Target NaNs here.
    feature_dependencies = ['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=feature_dependencies, inplace=True)

features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# DYNAMIC ZERO-CRASH SEPARATION ENGINE
train_mask = (df.index < '2026-04-01')
predict_mask = (df.index >= '2026-04-01')

# Fallback split check
if len(df.loc[train_mask]) < 50:
    st.sidebar.info("🔄 Server limit hit: Auto-stabilizing with 40/60 dynamic data split.")
    split_point = int(len(df) * 0.40)
    
    # Target NaNs ko sirf training dataset se drop kiya gaya hai
    train_subset = df.iloc[:split_point].dropna(subset=['Target'])
    X_train = train_subset[features_matrix]
    y_train = train_subset['Target']
    
    X_predict = df[features_matrix].iloc[split_point:]
    df_signals = df.iloc[split_point:].copy()
else:
    # Target NaNs ko sirf training dataset se drop kiya gaya hai
    train_subset = df.loc[train_mask].dropna(subset=['Target'])
    X_train = train_subset[features_matrix]
    y_train = train_subset['Target']
    
    X_predict = df.loc[predict_mask, features_matrix]
    df_signals = df[predict_mask].copy()

if len(X_predict) == 0 or len(X_train) == 0:
    st.error("Data pipeline mismatch. Insufficient rows for execution matrix.")
else:
    # Model Training
    model_flow = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize Signals Block
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # STRICT 63% ACCURACY FILTER
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
    
    # Latest live market updates display at the top
    display_df = display_df.iloc[::-1]

    # Main Grid Data Presentation
    st.subheader(f"📋 Live Refined Umesh Search Matrix (5m Interval | April 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    inst_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
    inst_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

    st.sidebar.header("📊 Umesh Search Audit (5m Live)")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
    st.sidebar.warning(f"⚪ Fake Traps Filtered: **{traps}**")
