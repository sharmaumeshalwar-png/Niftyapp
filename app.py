import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 50: 50/50 Strategy Engine", layout="wide")
st.title("⚡ Nifty 50 | 50% Learn - 50% Predict | Standalone Engine")

# =====================================================================
# MATHEMATICAL ENGINE
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0):
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p
    q, r = 0.001, 0.1
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Processing 2-Year Nifty Data..."):
    # 2 Years Nifty 50 Data
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    df = pd.DataFrame(index=raw_df.index)
    df['a_Close'] = raw_df['Close'].iloc[:, 0]
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Features
    df['Order_Imbalance'] = (df['a_Close'] - raw_df['Low'].iloc[:, 0]) / (raw_df['High'].iloc[:, 0] - raw_df['Low'].iloc[:, 0] + 1e-10)
    df['Body_Imbalance'] = (((raw_df['Open'].iloc[:, 0] + df['a_Close']) / 2) - raw_df['Low'].iloc[:, 0]) / (raw_df['High'].iloc[:, 0] - raw_df['Low'].iloc[:, 0] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Fixed 25-Candle Target
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# DYNAMIC SPLIT (50% Learn / 50% Predict)
# =====================================================================
split_idx = int(len(df) * 0.50)
features = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

# Training on first 50%
model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])

# Predicting on full dataset
probs = model.predict_proba(df[features])
df['Prob_Up'] = probs[:, 1]
df['Prob_Down'] = probs[:, 0]

# Accumulator Logic
accumulator = 0
scores_log = []
signals = []

for p_up, p_down in zip(df['Prob_Up'], df['Prob_Down']):
    if p_up >= 0.55: accumulator += 1
    elif p_down >= 0.55: accumulator -= 1
    accumulator = max(-5, min(5, accumulator))
    scores_log.append(accumulator)
    
    if accumulator >= 4: signals.append("🟢 STRONG BUY")
    elif accumulator <= -4: signals.append("🔴 STRONG SELL")
    else: signals.append(f"⚪ NEUTRAL ({accumulator})")

df['Accumulator_Score'] = scores_log
df['d_ML_Signal'] = signals

# Display
display_cols = ['a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'd_ML_Signal']
st.dataframe(df[display_cols].sort_index(ascending=False), use_container_width=True, height=600)
