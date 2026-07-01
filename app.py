import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Supertrend ML Bot", layout="wide")
st.title("⚡ Nifty 50 Super-Fast Supertrend ML Engine")
st.write("🎯 **Core Logic:** Trigger on 'c' Sign Flip ➡️ Cross-match with Highly Sensitive Supertrend (5, 1) ➡️ Early Signal Capture")

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

# HIGHLY SENSITIVE SUPERTREND LOGIC (Setting: Period=5, Multiplier=1 for Fastest Hint)
def calculate_fast_supertrend(df, period=5, multiplier=1):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # ATR Calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Basic Bands
    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    # Final Bands Initialization
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df)) # 1 for Up, -1 for Down
    
    for i in range(1, len(df)):
        # Upper Band
        if basic_ub.iloc[i] < final_ub[i-1] or close.iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Lower Band
        if basic_lb.iloc[i] > final_lb[i-1] or close.iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Direction & Supertrend Line
        if supertrend[i-1] == final_ub[i-1]:
            direction[i] = 1 if close.iloc[i] > final_ub[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] < final_lb[i] else 1
            
        supertrend[i] = final_lb[i] if direction[i] == 1 else final_ub[i]
        
    df['ST_Line'] = supertrend
    df['ST_Direction'] = direction
    return df

# Fetch Data for NIFTY 50 Index
with st.spinner("Initializing Fast Supertrend (5,1) & Matrix Engine..."):
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
    
    # Inject Fast Supertrend
    df = calculate_fast_supertrend(df, period=5, multiplier=1)
    
    # Vector Distance from Supertrend to see early shifts
    df['ST_Distance'] = df['a_Close'] - df['ST_Line']
    
    # Target Setup (3 Hours Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['ST_Line', 'ST_Distance', 'Target'], inplace=True)

# Clean Feature Grid focusing on 'c' and Supertrend Action
features_matrix = ['c_Combined', 'ST_Line', 'ST_Direction', 'ST_Distance']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train ONLY on crossover points to see Supertrend impact
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Supertrend alignment mismatch. Please restart application.")
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
    
    # 60% standard sensitive threshold matching
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.60) & (df_signals['ST_Direction'] == 1), 'd_ML_Signal'] = "🟢 FAST TREND UP (Early Hint)"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.60) & (df_signals['ST_Direction'] == -1), 'd_ML_Signal'] = "🔴 FAST TREND DOWN (Early Hint)"
    
    # Conflict filter (If sign change says UP but Supertrend says DOWN, it's a trap)
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ CONFLICT NOISE (Avoid Fake)"
    
    # Lock when no sign change happens
    df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

    # Clean display frame extraction
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Main Grid Data Presentation
    st.subheader(f"📋 Nifty 50 Supertrend Alignment Matrix (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Filter Counter Metrics
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    fast_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 FAST TREND UP (Early Hint)"])
    fast_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 FAST TREND DOWN (Early Hint)"])
    blocked_fakes = len(df_signals[df_signals['d_ML_Signal'] == "⚪ CONFLICT NOISE (Avoid Fake)"])

    st.sidebar.header("📊 Fast Supertrend Analytics")
    st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
    st.sidebar.write(f"🟢 Early Validated Buys: **{fast_buys}**")
    st.sidebar.write(f"🔴 Early Validated Sells: **{fast_sells}**")
    st.sidebar.warning(f"⚪ Mismatched traps Filtered: **{blocked_fakes}**")
