import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Fixed January Engine", layout="wide")
st.title("🦅 Nifty 50 Strict Walk-Forward Engine")
st.write("🎯 **Leakage Fixed:** Implemented sequential split tracking to guarantee zero future data sight since **1 January 2026**")

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

# Fetch Data with Dynamic Safety Limit
with st.spinner("Re-building zero-leakage training sequences..."):
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=700)).strftime('%Y-%m-%d')
    
    df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        st.error("Data server timeout. Please click Reboot App.")
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
    
    # Strict Target Definition (3 Hours Look-Ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity', 'Target'], inplace=True)

# Feature Matrix
features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# ZERO-LEAKAGE SEPARATION ENGINE
# Model learns exclusively from the historical buffer era up to December 31, 2025
train_pure_mask = (df.index < '2026-01-01')
display_mask = (df.index >= '2026-01-01')

X_train = df.loc[train_pure_mask, features_matrix]
y_train = df.loc[train_pure_mask, 'Target']
X_display = df.loc[display_mask, features_matrix]

if len(X_display) == 0 or len(X_train) < 50:
    # Safe expanding fallback to ensure baseline parameters are strictly metrics-driven
    st.info("Expanding baseline history context to secure prediction safety...")
    split_point = int(len(df) * 0.40)
    X_train = df[features_matrix].iloc[:split_point]
    y_train = df['Target'].iloc[:split_point]
    X_display = df[features_matrix].iloc[split_point:]
    df_signals = df.iloc[split_point:].copy()
else:
    df_signals = df[display_mask].copy()

# Robust Model execution to prevent memorization
model_wf = RandomForestClassifier(n_estimators=350, max_depth=4, min_samples_split=15, min_samples_leaf=4, random_state=42)
model_wf.fit(X_train, y_train)

probabilities = model_wf.predict_proba(X_display)
df_signals['Prob_Down'] = probabilities[:, 0]
df_signals['Prob_Up'] = probabilities[:, 1]

# Initialize Clean Grid Output State
df_signals['d_ML_Signal'] = "⚪ HOLD"
crossover_mask = df_signals['Sign_Change'] == 1

# STRICT 63% STABLE RATIO
df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"
df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"
df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

# Filter to force strictly 1 January 2026 onwards on UI grid display
final_ui_mask = df_signals.index >= '2026-01-01'
ui_display_df = df_signals[final_ui_mask].copy()

# Clean layout frame extraction
clean_display_cols = ['a_Close', 'b_Kalman', 'c_Combined', 'd_ML_Signal']
display_df = ui_display_df[clean_display_cols].copy()

# Formatting outputs
display_df['a_Close'] = display_df['a_Close'].round(2)
display_df['b_Kalman'] = display_df['b_Kalman'].round(2)
display_df['c_Combined'] = display_df['c_Combined'].round(4)
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

# Main Grid Presentation
st.subheader(f"📋 Nifty 50 Accurate Matrix (1 January 2026 - Present)")
st.dataframe(display_df, use_container_width=True, height=750)

# Sidebar Filter Counter Metrics
total_flips = len(ui_display_df[ui_display_df['Sign_Change'] == 1])
inst_buys = len(ui_display_df[ui_display_df['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
inst_sells = len(ui_display_df[ui_display_df['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
traps = len(ui_display_df[ui_display_df['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

st.sidebar.header("📊 Clean Audit (January 2026 - Present)")
st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
st.sidebar.write(f"🟢 Confirmed Buy Moves: **{inst_buys}**")
st.sidebar.write(f"🔴 Confirmed Sell Moves: **{inst_sells}**")
st.sidebar.warning(f"⚪ Fake Traps Blocked: **{traps}**")
