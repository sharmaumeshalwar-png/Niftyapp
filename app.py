import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="NIFTY50 Standalone 0.50 Engine", layout="wide")
st.title("⚡ NIFTY 50 Live 1-Hour Standalone Double Kalman [0.50 Engine]")
st.write("🎯 **Nifty Custom Setting:** Index Data (^NSEI) + Price Kalman + Fixed 25-Candle Target Window + Pure Raw Accumulator")

# =====================================================================
# MATHEMATICAL ENGINE (Same Kalman Logic)
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

with st.spinner("Aligning 25-Candle Double Kalman Nifty Microstructure..."):
    # Nifty 50 Ticker
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("Nifty data unavailable. Check market hours/internet.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Imbalance'] = (((df['Open'] + df['a_Close']) / 2) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# DYNAMIC SPLIT ENGINE
# =====================================================================
split_idx = int(len(df) * 0.50)
features = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])

probs = model.predict_proba(df.iloc[split_idx:][features])
df_predict = df.iloc[split_idx:].copy()
df_predict['Prob_Up'] = probs[:, 1]
df_predict['Prob_Down'] = probs[:, 0]

# Signal Logic
accumulator = 0
signals = []
for p_up, p_down in zip(df_predict['Prob_Up'], df_predict['Prob_Down']):
    if p_up >= 0.55: accumulator = min(5, accumulator + 1)
    elif p_down >= 0.55: accumulator = max(-5, accumulator - 1)
    
    if accumulator == 5: signals.append("🟢 STRONG BUY")
    elif accumulator == -5: signals.append("🔴 STRONG SELL")
    else: signals.append(f"⚪ NEUTRAL (Score: {accumulator})")

df_predict['d_ML_Signal'] = signals
df_predict['Accumulator_Score'] = accumulator

# Display
display_df = df_predict[['a_Close', 'Prob_Up', 'Accumulator_Score', 'd_ML_Signal']].sort_index(ascending=False)
st.dataframe(display_df.head(20), use_container_width=True)
