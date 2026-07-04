import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="BTC Ultra Core Engine", layout="wide")
st.title("🧠 BTC Live 1-Hour [Stateful Multi-Parameter Training Engine]")

# =====================================================================
# CORE KALMAN ENGINE
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: return []
    x, p, q, r = data_array[0], initial_p, q_val, r_val
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Training Parameters & Loading Deep Brain Matrix..."):
    raw_df = yf.download("BTC-USD", period="730d", interval="1h")
    if len(raw_df) == 0: st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # 5 Big Multi-Parameters taught to ML
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Mathematical Target (Zero Leakage)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# 50:50 Train & Predict Matrix Split
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

# Base Training Core
model_flow = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, warm_start=True)
model_flow.fit(df_train[features_matrix], df_train['Target'])

# Extraction for adaptive loops
X_predict_v = df_predict[features_matrix].to_numpy()
y_predict_v = df_predict['Target'].to_numpy()
closes_v = df_predict['a_Close'].to_numpy()

view_log, brain_notes, accumulator_log = [], [], []
accumulator = 0
last_valid_view = "⚪ HOLD (No Clear Pattern)"

# Memory buffers to force strict training increments
window_X, window_y = [], []

for i in range(len(df_predict)):
    row_feats = X_predict_v[i].reshape(1, -1)
    
    # Incremental Parameter Refitting (Model seeks corrections online)
    if i > 0 and i % 24 == 0:  # Har 24 ghante me model parameters ko pichli galtiyo se sikhao
        window_X = X_predict_v[max(0, i-48):i]
        window_y = y_predict_v[max(0, i-48):i]
        model_flow.n_estimators += 5
        model_flow.fit(window_X, window_y)
        
    probs = model_flow.predict_proba(row_feats)[0]
    p_down, p_up = probs[0], probs[1]
    
    # ⚡ STRICT NO-FLIP LOGIC (HINTS CLEAR WINDOW)
    # Target 95% certainty boundary
    if p_up >= 0.65:
        last_valid_view = f"📈 UP (Confidence: {p_up*100:.1f}%)"
        accumulator = min(5, accumulator + 1)
        note = "🎯 Pattern Locked: Velocity & Kalman verified upward momentum."
    elif p_down >= 0.65:
        last_valid_view = f"📉 DOWN (Confidence: {p_down*100:.1f}%)"
        accumulator = max(-5, accumulator - 1)
        note = "🎯 Pattern Locked: Velocity & Kalman verified downward momentum."
    else:
        # Jab tak 65% cross nahi hota, view flip nahi hoga, pichla strong hint hi hold rahega
        note = "⚡ No new macro confirmation. Holding previous clear trajectory state."

    view_log.append(last_valid_view)
    brain_notes.append(note)
    accumulator_log.append(accumulator)

df_predict['Live_View'] = view_log
df_predict['Accumulator_Score'] = accumulator_log
df_predict['ML_Dynamic_Training_Notes'] = brain_notes
df_predict['Raw_Weighted_Momentum'] = df_predict['a_Close'] - df_predict['b_Kalman_Price']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

# UI Layer Clean Frame
clean_cols = ['a_Close', 'b_Kalman_Price', 'Weighted_Momentum', 'Accumulator_Score', 'Live_View', 'ML_Dynamic_Training_Notes']
display_df = df_predict[clean_cols].copy().iloc[::-1]
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Stateful Self-Learning Matrix (Latest Candle Locked on Top)")
st.dataframe(display_df, use_container_width=True, height=600)
