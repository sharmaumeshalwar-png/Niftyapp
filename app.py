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

# Base Training Core (Removed warm_start to prevent dimension mismatch crashes)
model_flow = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
model_flow.fit(df_train[features_matrix], df_train['Target'])

# Extractions for adaptive loops
X_predict_v = df_predict[features_matrix].to_numpy()
closes_v = df_predict['a_Close'].to_numpy()

view_log, brain_notes, accumulator_log = [], [], []
accumulator = 0
last_valid_view = "⚪ HOLD (No Clear Pattern)"

# Active incremental memory arrays
accumulated_X = df_train[features_matrix].copy()
accumulated_y = df_train['Target'].copy()

for i in range(len(df_predict)):
    row_feats = X_predict_v[i].reshape(1, -1)
    
    # Safe Parameter Refitting: Mixed memory technique to retain both classes [0, 1]
    if i > 0 and i % 24 == 0:
        recent_chunk_X = df_predict[features_matrix].iloc[max(0, i-24):i]
        recent_chunk_y = df_predict['Target'].iloc[max(0, i-24):i]
        
        # Append latest market memory safely
        accumulated_X = pd.concat([accumulated_X, recent_chunk_X]).iloc[-len(df_train):]
        accumulated_y = pd.concat([accumulated_y, recent_chunk_y]).iloc[-len(df_train):]
        
        # Re-fit parameters safely without breaking shape constraints
        model_flow.fit(accumulated_X, accumulated_y)
        
    probs = model_flow.predict_proba(row_feats)[0]
    
    # Handle single-class edge cases gracefully if they ever appear
    if len(probs) == 2:
        p_down, p_up = probs[0], probs[1]
    else:
        p_down, p_up = (1.0, 0.0) if model_flow.classes_[0] == 0 else (0.0, 1.0)
    
    # STRICT TREND-LOCK FILTER (Aims for 95% target consistency)
    if p_up >= 0.63:
        last_valid_view = f"📈 UP (Confidence: {p_up*100:.1f}%)"
        accumulator = min(5, accumulator + 1)
        note = "🎯 Pattern Secured: Parameters tracking steady upward shift."
    elif p_down >= 0.63:
        last_valid_view = f"📉 DOWN (Confidence: {p_down*100:.1f}%)"
        accumulator = max(-5, accumulator - 1)
        note = "🎯 Pattern Secured: Parameters tracking steady downward shift."
    else:
        note = "⚡ Sideways mixing filtered. Locking previous highest conviction trajectory."

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
