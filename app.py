import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty VIX-Sync Engine", layout="wide")
st.title("⚡ Nifty 50 Live 1-Hour [VIX-Sync Accumulator Engine]")
st.write("🎯 **Nifty Custom Setting:** 50:50 Split | 25-Candle Lookback | 0.55 Probability | India VIX-Synced Accumulator")

# =====================================================================
# VIX FETCHING & MATH ENGINE
# =====================================================================
def get_vix_factor():
    try:
        vix_df = yf.download("^INDIAVIX", period="1d", interval="1h")
        if not vix_df.empty:
            current_vix = vix_df['Close'].iloc[-1].item()
            return current_vix / 15.0
        return 1.0
    except:
        return 1.0

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

with st.spinner("Aligning Nifty Microstructure & Syncing VIX..."):
    # Nifty 50 Data
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    vix_multiplier = get_vix_factor()
    
    if len(raw_df) == 0:
        st.error("Market Data Unavailable.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Imbalance'] = (((df['Open'] + df['a_Close']) / 2) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# DYNAMIC SPLIT & VIX-SYNCED ENGINE
# =====================================================================
split_idx = int(len(df) * 0.50)
features = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])

df_predict = df.iloc[split_idx:].copy()
probs = model.predict_proba(df_predict[features])
df_predict['Prob_Up'] = probs[:, 1]
df_predict['Prob_Down'] = probs[:, 0]

# Accumulator & VIX-Sync Logic
accumulator = 0
signals = []
scores_log = []
vix_adj_scores_log = []

for p_up, p_down in zip(df_predict['Prob_Up'], df_predict['Prob_Down']):
    if p_up >= 0.55: accumulator += 1
    elif p_down >= 0.55: accumulator -= 1
    
    # Clamping range
    accumulator = max(-5, min(5, accumulator))
    scores_log.append(accumulator)
    
    # VIX-Synced Score
    synced_score = int(accumulator * vix_multiplier)
    synced_score = max(-5, min(5, synced_score))
    vix_adj_scores_log.append(synced_score)
    
    if synced_score >= 4: signals.append("🟢 STRONG BUY (VIX-Synced)")
    elif synced_score <= -4: signals.append("🔴 STRONG SELL (VIX-Synced)")
    else: signals.append(f"⚪ NEUTRAL (Score: {synced_score})")

df_predict['d_ML_Signal'] = signals
df_predict['Accumulator_Score'] = scores_log
df_predict['VIX_Adjusted_Score'] = vix_adj_scores_log

# Display
display_df = df_predict[['a_Close', 'Prob_Up', 'Accumulator_Score', 'VIX_Adjusted_Score', 'd_ML_Signal']].sort_index(ascending=False)
st.dataframe(display_df.head(50), use_container_width=True, height=600)
