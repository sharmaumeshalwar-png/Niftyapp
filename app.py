import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Custom Regime Bot", layout="wide")
st.title("🛡️ Nifty BeES: Sign-Change & Volume/VWAP Engine")
st.write("🎯 **Logic:** Trigger ONLY when 'c' changes sign ➡️ Analyze Volume & VWAP ➡️ Generate 'd' Signal")

# =====================================================================
# MATHEMATICAL ENGINE (b = Kalman Filter with 0.001 Process Noise)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # 0.001 noise parameter specified by you
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

def calculate_vwap_native(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    # Price aur VWAP ka distance matrix jo ML seekhega
    df['VWAP_Distance'] = df['Close'] - df['VWAP']
    return df

# Fetch Data
with st.spinner("Building Sign-Change Core System..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download("NIFTYBEES.NS", start="2025-01-01", end=end_date, interval="1h")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    # a = Close Price
    df['a_Close'] = df['Close']
    
    # b = Kalman Filter (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # Calculate VWAP
    df = calculate_vwap_native(df)
    
    # c_Combined (Difference between raw and smooth price)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # CRITICAL TRIGGER: Check if 'c' changes sign (Positive to Negative OR Negative to Positive)
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Target definition (Forward Look-ahead Trend)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VWAP', 'Target'], inplace=True)

# Feature selection: ONLY Volume and VWAP Analysis when c changes sign
features_matrix = ['Volume', 'VWAP_Distance']

# Mask for Train (2025) and Test (2026 - Today)
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Filter training data so ML ONLY learns from candles where sign change occurred
train_sign_change = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_change, features_matrix]
y_train = df.loc[train_sign_change, 'Target']

X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Sign change calculations data error. Please reboot application.")
else:
    # Train ML only on Volume + VWAP context during crossovers
    model_d = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_d.fit(X_train, y_train)

    # Predict probabilities for all test slots
    probabilities = model_d.predict_proba(X_test_all)
    
    df_signals = df[test_mask].copy()
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Rule Setup
    df_signals['d_ML_Signal'] = "⚪ HOLD (No Cross)"
    
    # Condition: ONLY evaluate when Sign_Change == 1, else keep HOLD
    # High surety threshold set at 60% for isolated crossover points
    crossover_mask = df_signals['Sign_Change'] == 1
    
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.60), 'd_ML_Signal'] = "🟢 SURE BUY (Volume Breakout)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.60), 'd_ML_Signal'] = "🔴 SURE SELL (Volume Breakdown)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] < 0.60) & (df_signals['Prob_Down'] < 0.60), 'd_ML_Signal'] = "⚪ WAIT (Low Volume Break)"

    # Clean display frame formation
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'Volume', 'VWAP', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Round values for clear visibility
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df['VWAP'] = display_df['VWAP'].round(2)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Main Screen Data Render
    st.subheader(f"📋 Crossover & Volume Execution Table: 1 Jan 2026 to Present")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Filter Stats
    total_crosses = len(df_signals[df_signals['Sign_Change'] == 1])
    active_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 SURE BUY (Volume Breakout)"])
    active_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 SURE SELL (Volume Breakdown)"])
    
    st.sidebar.header("📊 Regime Stats (2026)")
    st.sidebar.write(f"Total Sign Changes Checked: **{total_crosses}**")
    st.sidebar.write(f"🟢 Confirmed Buy Breaks: **{active_buys}**")
    st.sidebar.write(f"🔴 Confirmed Sell Breaks: **{active_sells}**")
    st.sidebar.info("Note: Jab 'c' apna sign badlega, tabhi ML check karega ki volume aur VWAP me taakat hai ya nahi. Agar volume thanda hoga, to 'WAIT' dikhayega.")
