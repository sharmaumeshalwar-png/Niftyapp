import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty News-Adaptive Engine", layout="wide")
st.title("🦅 Nifty 50 News-Adaptive ML Engine (VIX Integrated)")
st.write("🎯 **Logic:** Trigger on 'c' Sign Flip ➡️ Corelate with External News Sentiment (India VIX Shock Indicator)")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
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

# Fetch Data for NIFTY 50 and External Source (India VIX)
with st.spinner("Integrating external market news and VIX sentiment engine..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 1. Fetch Nifty Data
    df_nifty = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    # 2. Fetch India VIX (The Fear/News Index tracker for Indian Markets)
    df_vix = yf.download("^INDIAVIX", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df_nifty.columns, pd.MultiIndex): df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)

    # Align Data Frames on common indices
    df = df_nifty[['High', 'Low', 'Close']].copy()
    df['a_Close'] = df['Close']
    df['VIX_Close'] = df_vix['Close']
    df.ffill(inplace=True)  # Handle timing mismatches if any
    
    # b = Kalman Filter (0.001)
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    
    # c = Price vs Kalman Gap
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # EXTERNAL NEWS FEATURE TRACKING
    df['VIX_Velocity'] = df['VIX_Close'].diff(1)  # High velocity means heavy incoming news
    df['VIX_Shock'] = np.where(df['VIX_Velocity'] > df['VIX_Velocity'].rolling(10).std() * 1.5, 1, 0) # Sudden Event Alert
    
    # Retain structural zone indicators
    df['Buying_Zone'] = df['Low'].rolling(window=20).min()
    df['Selling_Zone'] = df['High'].rolling(window=20).max()
    df['Dist_From_Buy_Zone'] = df['a_Close'] - df['Buying_Zone']
    df['Dist_From_Sell_Zone'] = df['Selling_Zone'] - df['a_Close']
    
    # Target (Forward Trend Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['VIX_Shock', 'Dist_From_Buy_Zone', 'Target'], inplace=True)

# Feature Matrix now includes External News Sentiment Metrics
features_matrix = ['VIX_Close', 'VIX_Velocity', 'VIX_Shock', 'Dist_From_Buy_Zone', 'Dist_From_Sell_Zone', 'c_Combined']

train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

# Train on crossovers under news/VIX conditions
train_sign_moments = train_mask & (df['Sign_Change'] == 1)

X_train = df.loc[train_sign_moments, features_matrix]
y_train = df.loc[train_sign_moments, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0:
    st.error("Data pipeline mismatch. Please hard reboot.")
else:
    # Model Setup
    model_d = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    model_d.fit(X_train, y_train)

    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Process Signals
    df_signals['d_ML_Signal'] = "⚪ HOLD"
    crossover_mask = df_signals['Sign_Change'] == 1
    
    # Strict 68% verification filter to absorb high volatility shocks
    df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.68), 'd_ML_Signal'] = "🟢 NEWS VALIDATED BUY"
    df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.68), 'd_ML_Signal'] = "🔴 NEWS VALIDATED SELL"
    
    # Explicitly catch sudden high-risk spikes and lock them to avoid trading bad events
    df_signals.loc[crossover_mask & (df_signals['VIX_Shock'] == 1), 'd_ML_Signal'] = "⚠️ SUDDEN NEWS EVENT (No Trade)"
    df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD") & (df_signals['Prob_Up'] < 0.68), 'd_ML_Signal'] = "⚪ FALSE FLIP (Avoid)"

    # Filter clean output layout
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating values
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Main Screen Data View
    st.subheader(f"📋 News-Integrated Structural Table (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Metric System
    total_flips = len(df_signals[df_signals['Sign_Change'] == 1])
    news_blocks = len(df_signals[df_signals['d_ML_Signal'] == "⚠️ SUDDEN NEWS EVENT (No Trade)"])
    real_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 NEWS VALIDATED BUY"])
    real_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 NEWS VALIDATED SELL"])

    st.sidebar.header("📊 Sentiment Intelligence")
    st.sidebar.write(f"Total Sign Changes: **{total_flips}**")
    st.sidebar.warning(f"💥 Sudden News Spikes Deflected: **{news_blocks}**")
    st.sidebar.write(f"🟢 News Validated Buys: **{real_buys}**")
    st.sidebar.write(f"🔴 News Validated Sells: **{real_sells}**")
