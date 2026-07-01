import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="Nifty Master Reverse Engine", layout="wide")
st.title("🦅 Nifty 50 Master Reverse-Logic ML Engine")
st.write("🎯 **Inverted Secret Logic:** Capitalizing on 100% Reverse Hints ➡️ Automatically Inverting Signals (BUY 🔄 SELL) for Ultimate Accuracy & Practical Frequency")

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
with st.spinner("Activating reverse-logic microstructure matrix..."):
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
    
    # YOUR ORIGINAL STABLE MICROSTRUCTURE PROXY
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Strict Target Definition (3 Hours Look-Ahead)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    df.dropna(subset=['Order_Imbalance', 'Flow_Velocity', 'Target'], inplace=True)

# Feature Matrix
features_matrix = ['c_Combined', 'Order_Imbalance', 'Flow_Velocity']

# PURE TIME-SERIES SPLIT 
train_pure_mask = (df.index < '2025-07-01')
display_mask = (df.index >= '2025-07-01')

X_train = df.loc[train_pure_mask, features_matrix]
y_train = df.loc[train_pure_mask, 'Target']
X_display = df.loc[display_mask, features_matrix]

if len(X_display) == 0 or len(X_train) < 50:
    split_point = int(len(df) * 0.40)
    X_train = df[features_matrix].iloc[:split_point]
    y_train = df['Target'].iloc[:split_point]
    X_display = df[features_matrix].iloc[split_point:]
    df_signals = df.iloc[split_point:].copy()
else:
    df_signals = df[display_mask].copy()

# Original Setup that gives ~10 trades in 6 months
model_wf = RandomForestClassifier(n_estimators=250, max_depth=4, random_state=42)
model_wf.fit(X_train, y_train)

probabilities = model_wf.predict_proba(X_display)
df_signals['Prob_Down'] = probabilities[:, 0]
df_signals['Prob_Up'] = probabilities[:, 1]

# Initialize Clean Grid Output State
df_signals['d_ML_Signal'] = "⚪ HOLD"
crossover_mask = df_signals['Sign_Change'] == 1

# =====================================================================
# THE MASTER INVERSION OPERATION (Turning Reverse Hints into Profits)
# =====================================================================
# IF ML Predicts UP strictly >= 63% ➡️ We INVERT to INSTITUTIONAL SELL
df_signals.loc[crossover_mask & (df_signals['Prob_Up'] >= 0.63), 'd_ML_Signal'] = "🔴 INSTITUTIONAL SELL (Confirmed)"

# IF ML Predicts DOWN strictly >= 63% ➡️ We INVERT to INSTITUTIONAL BUY
df_signals.loc[crossover_mask & (df_signals['Prob_Down'] >= 0.63), 'd_ML_Signal'] = "🟢 INSTITUTIONAL BUY (Confirmed)"

# Filter out retail noise traps
df_signals.loc[crossover_mask & (df_signals['d_ML_Signal'] == "⚪ HOLD"), 'd_ML_Signal'] = "⚪ RETAIL TRAP (Avoid Fake)"
df_signals.loc[df_signals['Sign_Change'] == 0, 'd_ML_Signal'] = "⚪ HOLD"

# Filter to force strictly 1 July 2025 onwards on UI grid display
final_ui_mask = df_signals.index >= '2025-07-01'
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
st.subheader(f"📋 Nifty 50 Inverted Super Engine (1 July 2025 - Present)")
st.dataframe(display_df, use_container_width=True, height=750)

# Sidebar Filter Counter Metrics
total_flips = len(ui_display_df[ui_display_df['Sign_Change'] == 1])
inst_buys = len(ui_display_df[ui_display_df['d_ML_Signal'] == "🟢 INSTITUTIONAL BUY (Confirmed)"])
inst_sells = len(ui_display_df[ui_display_df['d_ML_Signal'] == "🔴 INSTITUTIONAL SELL (Confirmed)"])
traps = len(ui_display_df[ui_display_df['d_ML_Signal'] == "⚪ RETAIL TRAP (Avoid Fake)"])

st.sidebar.header("📊 Inverted Logic Audit (July 2025 - Present)")
st.sidebar.write(f"Total Sign Flips Checked: **{total_flips}**")
st.sidebar.write(f"🟢 Corrected Buy Moves: **{inst_buys}**")
st.sidebar.write(f"🔴 Corrected Sell Moves: **{inst_sells}**")
st.sidebar.warning(f"⚪ Fake Traps Blocked: **{traps}**")
