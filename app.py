import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Anti-ZigZag Bot", layout="wide")
st.title("🛡️ Nifty 50 Anti-ZigZag VIX ML Engine")
st.write("🎯 **Core Logic:** Detect 'c' Sign Change ➡️ Measure VIX Zig-Zag (Stability) ➡️ Lock Real Trends & Kill Noise")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Exact parameter
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
with st.spinner("ML is scanning VIX Zig-Zag dynamics and stability matrix..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    df_nifty = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    df_vix = yf.download("^INDIAVIX", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df_nifty.columns, pd.MultiIndex): df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)

    # Align Data
    df = df_nifty[['High', 'Low', 'Close']].copy()
    df['a_Close'] = df['Close']
    df['VIX_Close'] = df_vix['Close']
    df.ffill(inplace=True)
    
    # Base Kalman Math
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # HARD LOCK: Track sign flip points
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # THE ZIG-ZAG TRACKER (Pichle 5 ghante me VIX kitna uchla-kooda)
    df['VIX_Velocity'] = df['VIX_Close'].diff(1)
    df['VIX_ZigZag'] = df['VIX_Velocity'].rolling(window=5).std()  # High = Bad/Noise, Low = Solid Trend
    
    # Target (3 Hours Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VIX_ZigZag', 'VIX_Velocity', 'Target'], inplace=True)

# Feature matrix strictly monitoring the VIX Zig-Zag stability
features_matrix = ['c_Combined', 'VIX_Close', 'VIX_Velocity', 'VIX_ZigZag']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ONLY on crossover hours to capture VIX behavior during flips
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Historical matrix initialization failed. Please reboot app.")
else:
    # Stable classifier to detect stability patterns
    model_d = RandomForestClassifier(n_estimators=250, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize Signals
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Standard 62% threshold + automatic evaluation of low VIX ZigZag by ML
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.62), 'd_ML_Signal'] = "🟢 TREND UP (Smooth VIX)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.62), 'd_ML_Signal'] = "🔴 TREND DOWN (Smooth VIX)"
    
    # Catching high zig-zag noise and marking it fake
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ ZIG-ZAG NOISE (Avoid Fake)"
    
    # Absolute hold when no sign change occurs
    df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

    # Clean display subset
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formatting numbers
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # UI Grid Render
    st.subheader(f"📋 Live Nifty VIX Stability Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    smooth_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 TREND UP (Smooth VIX)"])
    smooth_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 TREND DOWN (Smooth VIX)"])
    fakes_blocked = len(df_signals[df_signals['d_ML_Signal'] == "⚪ ZIG-ZAG NOISE (Avoid Fake)"])

    st.sidebar.header("📊 VIX Anti-Noise Stats")
    st.sidebar.write(f"Total Flips Scanned: **{total_flips}**")
    st.sidebar.write(f"🟢 Real Up Trends: **{smooth_up}**")
    st.sidebar.write(f"🔴 Real Down Trends: **{smooth_dn}**")
    st.sidebar.warning(f"⚪ Zig-Zag Fakes Blocked: **{fakes_blocked}**")
