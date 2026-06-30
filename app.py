import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Zone ML Bot", layout="wide")
st.title("🛡️ Nifty 50 Smart Zone-Detection ML Engine")
st.write("🎯 **Core Logic:** Trigger on 'c' Sign Flip ➡️ Analyze Proximity to 20-Hour Buying/Selling Zones ➡️ High-Surety Signal")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Exact 0.001 constraint
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
with st.spinner("ML is scanning Buying and Selling Zones in history..."):
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
    
    # SYSTEM ZONES: 20-Hour Lookback Supply and Demand Areas
    df['Selling_Zone'] = df['High'].rolling(window=20).max()  # Recent Major Resistance
    df['Buying_Zone'] = df['Low'].rolling(window=20).min()    # Recent Major Support
    
    # Distance from Zones (How close is Nifty to the danger or opportunity areas?)
    df['Dist_From_Buy_Zone'] = df['a_Close'] - df['Buying_Zone']
    df['Dist_From_Sell_Zone'] = df['Selling_Zone'] - df['a_Close']
    
    # Physics parameters from previous setup retained for accuracy
    df['Price_Velocity'] = df['a_Close'].diff(1)
    df['Elasticity'] = df['Price_Velocity'].rolling(window=3).sum()
    
    # Target definition (Forward Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Dist_From_Buy_Zone', 'Dist_From_Sell_Zone', 'Elasticity', 'Target'], inplace=True)

# Features Matrix built to feed Zone Distances to ML
features_matrix = ['Dist_From_Buy_Zone', 'Dist_From_Sell_Zone', 'Price_Velocity', 'Elasticity', 'c_Combined']

# Train (2025) and Test (2026 - Today) Masks
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ML ONLY during sign changes to study zone reactions
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Zone mapping configuration error. Please reboot application.")
else:
    # Model Setup
    model_d = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Initialize execution block
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # 65% surety threshold combined with zone coordinates
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.65), 'd_ML_Signal'] = "🟢 BUY ZONE BREAKOUT (Sure)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.65), 'd_ML_Signal'] = "🔴 SELL ZONE BREAKDOWN (Sure)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] < 0.65) & (df_signals['Prob_Down'] < 0.65), 'd_ML_Signal'] = "⚪ FALSE ZONE FLIP (Avoid)"

    # Clean display frame
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Web UI Layout Render
    st.subheader(f"📋 Nifty 50 Structural Zone Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    real_trends_up = len(df_signals[df_signals['d_ML_Signal'] == "🟢 BUY ZONE BREAKOUT (Sure)"])
    real_trends_dn = len(df_signals[df_signals['d_ML_Signal'] == "🔴 SELL ZONE BREAKDOWN (Sure)"])
    false_flips = len(df_signals[df_signals['d_ML_Signal'] == "⚪ FALSE ZONE FLIP (Avoid)"])

    st.sidebar.header("📊 Zone Engine Analytics")
    st.sidebar.write(f"Total Sign Changes: **{total_flips}**")
    st.sidebar.write(f"🟢 Confirmed Buy Zone Trades: **{real_trends_up}**")
    st.sidebar.write(f"🔴 Confirmed Sell Zone Trades: **{real_trends_dn}**")
    st.sidebar.write(f"⚪ No-Zone/Shaky Fakes: **{false_flips}**")
