import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Fibonacci ML", layout="wide")
st.title("🦅 Nifty 50 Fibonacci Hybrid ML Engine (2026 Live Predict)")
st.write("🎯 **Core Logic:** Train on 2025 Data ➡️ Calculate 24h Rolling Fibonacci Levels ➡️ Predict 2026 Live Signals via Golden Ratio Alignment")

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

# Fetch Data
with st.spinner("Calculating dynamic Fibonacci levels and segmenting 2025-2026 timelines..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    # Fetching from late 2024 to ensure full indicators available on 1 Jan 2025
    df = yf.download("^NSEI", start="2024-11-15", end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("Data engine timeout. Please click Reboot App.")
        st.stop()

    # Base Pricing Vectors
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # Strict Sign Flip Lock
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # =====================================================================
    # DYNAMIC ROLLING FIBONACCI LEVELS (24 Trading Hours Window)
    # =====================================================================
    df['Roll_High'] = df['High'].rolling(window=24).max()
    df['Roll_Low'] = df['Low'].rolling(window=24).min()
    df['Range'] = df['Roll_High'] - df['Roll_Low'] + 1e-10
    
    # Fibonacci Retracement Levels
    df['Fib_236'] = df['Roll_High'] - (df['Range'] * 0.236)
    df['Fib_382'] = df['Roll_High'] - (df['Range'] * 0.382)
    df['Fib_500'] = df['Roll_High'] - (df['Range'] * 0.500)
    df['Fib_618'] = df['Roll_High'] - (df['Range'] * 0.618)
    
    # Distance from key institutional Fibonacci zones
    df['Fib_50_Dist'] = df['a_Close'] - df['Fib_500']
    df['Fib_618_Dist'] = df['a_Close'] - df['Fib_618']
    
    # Microstructure helper
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Setup (3 Hours Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Fib_50_Dist', 'Fib_618_Dist', 'Flow_Velocity', 'Target'], inplace=True)

# Feature Matrix for ML
features_matrix = ['c_Combined', 'Fib_50_Dist', 'Fib_618_Dist', 'Flow_Velocity']

# STRICT SEPARATION OF TIME LOGIC
train_2025_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
predict_2026_mask = (df.index >= '2026-01-01')

X_train = df.loc[train_2025_mask, features_matrix]
y_train = df.loc[train_2025_mask, 'Target']
X_predict = df.loc[predict_2026_mask, features_matrix]

if len(X_predict) == 0:
    st.warning("Prediction window empty. Synchronizing server timeline constraints.")
else:
    # Model Setup
    model_fib = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=6, random_state=42)
    model_fib.fit(X_train, y_train)

    probabilities = model_fib.predict_proba(X_predict)
    df_signals = df[predict_2026_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize Clean Signal Output
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # 58% optimized threshold combined with Fibonacci distance intelligence
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.58), 'd_ML_Signal'] = "🟢 FIBONACCI BUY (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.58), 'd_ML_Signal'] = "🔴 FIBONACCI SELL (Confirmed)"
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ FIB TRAP (Avoid Fake)"
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
    st.subheader(f"📋 Live Nifty 50 Fibonacci Predictions (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=800)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    fib_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 FIBONACCI BUY (Confirmed)"])
    fib_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 FIBONACCI SELL (Confirmed)"])
    traps = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FIB TRAP (Avoid Fake)"])

    st.sidebar.header("📊 2026 Prediction Audit")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Real Fibonacci Buys: **{fib_buys}**")
    st.sidebar.write(f"🔴 Real Fibonacci Sells: **{fib_sells}**")
    st.sidebar.warning(f"⚪ Fake Fib Traps Blocked: **{traps}**")
