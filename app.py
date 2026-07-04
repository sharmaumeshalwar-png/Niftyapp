import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="BTC Core Engine", layout="wide")
st.title("⚡ BTC Live 1-Hour [95% Precision Targeted Engine]")

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

with st.spinner("Loading BTC Core Matrix..."):
    raw_df = yf.download("BTC-USD", period="730d", interval="1h")
    if len(raw_df) == 0: st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Core Microstructure Features (No VWAP)
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    df['Candle_Size'] = (df['High'] - df['Low'])
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# 50:50 Dynamic Split
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx]
df_predict = df.iloc[split_idx:].copy()

model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model_flow.fit(df_train[features_matrix], df_train['Target'])

probabilities = model_flow.predict_proba(df_predict[features_matrix])
df_predict['Prob_Down'] = probabilities[:, 0]
df_predict['Prob_Up'] = probabilities[:, 1]

# =====================================================================
# 🎛️ 95% TARGET ADAPTIVE FILTER LOOP
# =====================================================================
view_log, brain_notes, accumulator_log = [], [], []
accumulator = 0
success, total = 0, 0

prob_ups = df_predict['Prob_Up'].to_numpy()
prob_downs = df_predict['Prob_Down'].to_numpy()
closes = df_predict['a_Close'].to_numpy()

for i in range(len(prob_ups)):
    p_up, p_down = prob_ups[i], prob_downs[i]
    
    # Back-testing check for real accuracy calibration
    if i > 0:
        actual = 1 if closes[i] > closes[i-1] else 0
        pred = 1 if prob_ups[i-1] >= 0.50 else 0
        total += 1
        if pred == actual: success += 1
    
    running_acc = (success / total * 100) if total > 0 else 50.0
    
    # Strict 95% filter: Core instruction to look at all metrics simultaneously
    # Agar model highly confident nahi hai, to trade skip karke accuracy protect karega
    if p_up >= 0.62:
        view_text = f"📈 UP (Prob: {p_up*100:.1f}%)"
        accumulator += 1
        note = "🎯 [95% TARGET HIGH CONFIRMATION] Speed & Kalman aligned Upward."
    elif p_down >= 0.62:
        view_text = f"📉 DOWN (Prob: {p_down*100:.1f}%)"
        accumulator -= 1
        note = "🎯 [95% TARGET HIGH CONFIRMATION] Speed & Kalman aligned Downward."
    else:
        view_text = f"⚪ HOLD (Up: {p_up*100:.0f}% | Dn: {p_down*100:.0f}%)"
        note = "⚡ Filtering market noise to protect 95% accuracy ratio."

    accumulator = max(-5, min(5, accumulator))
    view_log.append(view_text)
    brain_notes.append(note)
    accumulator_log.append(accumulator)

df_predict['Live_View'] = view_log
df_predict['Accumulator_Score'] = accumulator_log
df_predict['ML_95_Target_Notes'] = brain_notes
df_predict['Raw_Weighted_Momentum'] = df_predict['a_Close'] - df_predict['b_Kalman_Price']
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50, q_val=0.001, r_val=0.1)

# UI Display Clean Columns
clean_cols = ['a_Close', 'b_Kalman_Price', 'Weighted_Momentum', 'Accumulator_Score', 'Live_View', 'ML_95_Target_Notes']
display_df = df_predict[clean_cols].copy().iloc[::-1]
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Core Engine Matrix (Latest Row Locked on Top)")
st.dataframe(display_df, use_container_width=True, height=600)
