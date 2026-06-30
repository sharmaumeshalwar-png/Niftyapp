import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Advanced Blend Bot", layout="wide")
st.title("🦅 Nifty 50 Continuous Matrix Blend Engine (c + VIX Fusion)")
st.write("🎯 **Core Logic:** Multi-Dimensional Pattern Recognition — ML permanently analyzes the interaction of 'c' and India VIX")

# =====================================================================
# MATHEMATICAL SMOOTHING (b = Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    x = price_array[0]
    p = 50.0  
    q = 0.001  # Your exact 0.001 parameter
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
with st.spinner("ML is blending 'c' and India VIX into a unified historical matrix..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Download Core Assets
    df_nifty = yf.download("^NSEI", start="2025-01-01", end=end_date, interval="1h")
    df_vix = yf.download("^INDIAVIX", start="2025-01-01", end=end_date, interval="1h")
    
    if isinstance(df_nifty.columns, pd.MultiIndex): df_nifty.columns = df_nifty.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)

    # Sync and Align Data
    df = df_nifty[['High', 'Low', 'Close']].copy()
    df['a_Close'] = df['Close']
    df['VIX_Close'] = df_vix['Close']
    df.ffill(inplace=True)
    
    # Calculate Core Variables
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    # ADVANCED FUSION FEATURES (Where ML finds the real solid hints)
    df['VIX_Velocity'] = df['VIX_Close'].diff(1)
    df['Price_Velocity'] = df['a_Close'].diff(1)
    
    # 1. c and VIX Interaction Ratio (Hidden Force metric)
    df['c_VIX_Ratio'] = df['c_Combined'] / (df['VIX_Close'] + 1e-10)
    
    # 2. Divergence Factor (Are price and fear moving in opposite or same directions?)
    df['Divergence_Factor'] = df['Price_Velocity'] * df['VIX_Velocity']
    
    # 3. Supply/Demand Zones for boundaries
    df['Buying_Zone'] = df['Low'].rolling(window=20).min()
    df['Selling_Zone'] = df['High'].rolling(window=20).max()
    df['Dist_From_Buy_Zone'] = df['a_Close'] - df['Buying_Zone']
    df['Dist_From_Sell_Zone'] = df['Selling_Zone'] - df['a_Close']
    
    # Target definition (3-Hour Forward Trend Look-ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['c_VIX_Ratio', 'Divergence_Factor', 'Dist_From_Buy_Zone', 'Target'], inplace=True)

# Continuous Feature Space (No sign change locks, pure raw blend vectors)
features_matrix = ['c_Combined', 'VIX_Close', 'VIX_Velocity', 'c_VIX_Ratio', 'Divergence_Factor', 'Dist_From_Buy_Zone', 'Dist_From_Sell_Zone']

# Splitting Data Frames
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train = df.loc[train_mask, features_matrix]
y_train = df.loc[train_mask, 'Target']
X_test_all = df.loc[test_mask, features_matrix]

if len(X_train) == 0 or len(X_test_all) == 0:
    st.error("Data pipeline processing error. Please reboot application.")
else:
    # Deep Forest architecture to decode continuous correlations
    model_d = RandomForestClassifier(n_estimators=350, max_depth=6, min_samples_split=8, random_state=42)
    model_d.fit(X_train, y_train)

    # Compute continuous math probabilities
    probabilities = model_d.predict_proba(X_test_all)
    df_signals = df[test_mask].copy()
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # Rule Definition Layout
    # Strict 75% filter on the continuous blend space to catch only rock-solid hints
    df_signals['d_ML_Signal'] = "⚪ HOLD / NO PATTERN"
    
    df_signals.loc[df_signals['Prob_Up'] >= 0.75, 'd_ML_Signal'] = "🟢 SOLID BUY (c + VIX Confirmed)"
    df_signals.loc[df_signals['Prob_Down'] >= 0.75, 'd_ML_Signal'] = "🔴 SOLID SELL (c + VIX Confirmed)"
    
    # News protection rule remains embedded natively
    df_signals.loc[df_signals['VIX_Velocity'] > df_signals['VIX_Velocity'].rolling(10).std() * 2, 'd_ML_Signal'] = "⚠️ SHOCK EVENT (System Locked)"

    # Clean display frame extraction
    clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
    display_df = df_signals[clean_display_cols].copy()

    # Formating outputs
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
    display_df['c_Combined'] = display_df['c_Combined'].round(4)
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')

    # Main Grid Data Presentation
    st.subheader(f"📋 Live Continuous Execution Grid (1 Jan 2026 - Present)")
    st.dataframe(display_df, use_container_width=True, height=750)

    # Sidebar Metric Block
    total_hours = len(df_signals)
    solid_buys = len(df_signals[df_signals['d_ML_Signal'] == "🟢 SOLID BUY (c + VIX Confirmed)"])
    solid_sells = len(df_signals[df_signals['d_ML_Signal'] == "🔴 SOLID SELL (c + VIX Confirmed)"])
    shocks = len(df_signals[df_signals['d_ML_Signal'] == "⚠️ SHOCK EVENT (System Locked)"])

    st.sidebar.header("📊 Advanced Intelligence Stats")
    st.sidebar.write(f"Total Market Hours Scanned: **{total_hours}**")
    st.sidebar.write(f"🟢 High-Confidence Buys: **{solid_buys}**")
    st.sidebar.write(f"🔴 High-Confidence Sells: **{solid_sells}**")
    st.sidebar.warning(f"⚠️ High-Risk News Events Deflected: **{shocks}**")
    st.sidebar.info("System ab kisi crossover ka wait nahi karta. Yeh pure time 'c' aur VIX ke mathematical ratio aur trends ko map karke solid positional direction nikalta hai.")
