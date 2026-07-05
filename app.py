import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty 2Y Full-View Engine", layout="wide")
st.title("⚡ Nifty 50 Live 2-Year Full-View Engine")
st.write("🎯 **Full Data View:** 2 Years | 50:50 Split Logic | VIX-Sync Enabled")

# =====================================================================
# MATH ENGINE & VIX SYNC
# =====================================================================
def get_vix_factor():
    try:
        vix_df = yf.download("^INDIAVIX", period="1d", interval="1h")
        return vix_df['Close'].iloc[-1].item() / 15.0 if not vix_df.empty else 1.0
    except: return 1.0

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

with st.spinner("Loading & Processing 2-Year Data..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    vix_multiplier = get_vix_factor()
    
    df = pd.DataFrame(index=raw_df.index)
    df['a_Close'] = raw_df['Close'].iloc[:, 0] if isinstance(raw_df['Close'], pd.DataFrame) else raw_df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Features
    df['Order_Imbalance'] = (df['a_Close'] - raw_df['Low'].iloc[:, 0]) / (raw_df['High'].iloc[:, 0] - raw_df['Low'].iloc[:, 0] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# MODEL & SIGNAL LOGIC
# =====================================================================
split_idx = int(len(df) * 0.50)
features = ['c_Combined', 'Order_Imbalance', 'Normalized_Gap']

model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(df.iloc[:split_idx][features], df.iloc[:split_idx]['Target'])

# Prediction
probs = model.predict_proba(df[features]) # Poora data predict kiya
df['Prob_Up'] = probs[:, 1]
df['Prob_Down'] = probs[:, 0]

# Accumulator loop
accumulator = 0
vix_adj_scores = []
signals = []

for p_up, p_down in zip(df['Prob_Up'], df['Prob_Down']):
    if p_up >= 0.55: accumulator += 1
    elif p_down >= 0.55: accumulator -= 1
    accumulator = max(-5, min(5, accumulator))
    
    synced = int(accumulator * vix_multiplier)
    synced = max(-5, min(5, synced))
    vix_adj_scores.append(synced)
    
    if synced >= 4: signals.append("🟢 STRONG BUY")
    elif synced <= -4: signals.append("🔴 STRONG SELL")
    else: signals.append(f"⚪ NEUTRAL ({synced})")

df['VIX_Adjusted_Score'] = vix_adj_scores
df['d_ML_Signal'] = signals

# Display
st.dataframe(df[['a_Close', 'VIX_Adjusted_Score', 'd_ML_Signal']].sort_index(ascending=False), use_container_width=True, height=600)
