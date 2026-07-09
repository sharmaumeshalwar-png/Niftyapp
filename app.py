import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Advanced Discovery Engine", layout="wide")
st.title("🚀 Nifty 50 Advanced Multi-Layer Discovery Engine")

# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0):
    if len(data_array) == 0: return []
    x = data_array[0]
    p = initial_p
    q = 0.0001
    r = 2.5
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0: return []
    x = data_array[0]
    p = 1.0
    q = 0.05
    r = 0.2
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# DATA & INTEGRATED DISCOVERY ENGINE
# =====================================================================
with st.spinner("Merging Volatility, Momentum & Probability Matrices..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(index=raw_df.index)
    df['Close'] = raw_df['Close'].ffill()
    df['High'] = raw_df['High'].ffill()
    df['Low'] = raw_df['Low'].ffill()
    
    # ATR Calculation for Volatility Integration
    df['HL_Range'] = df['High'] - df['Low']
    df['ATR'] = df['HL_Range'].rolling(14).mean()
    
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Gap'] = df['a_Close'] - df['b_Kalman']
    
    # ML Features (Learning the volatility regime)
    df['Normalized_Gap'] = df['c_Gap'] / (df['ATR'] + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# Training Model
split_idx = int(len(df) * 0.50)
train_df, predict_df = df.iloc[:split_idx], df.iloc[split_idx:].copy()
features = ['c_Gap', 'Normalized_Gap', 'ATR']
model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42).fit(train_df[features], train_df['Target'])

# Probability Merge
predict_df['Prob_Up'] = model.predict_proba(predict_df[features])[:, 1]

# Multi-Layer Momentum Calculation
predict_df['Weighted_Momentum'] = apply_kalman_filter_custom(predict_df['c_Gap'].values, initial_p=0.50)
predict_df['Step_Momentum'] = np.round(apply_non_linear_kalman_momentum(predict_df['Weighted_Momentum'].values))

# FINAL INTEGRATION: Discovery Score = (Prob * Weighted) / ATR
predict_df['Discovery_Score'] = (predict_df['Prob_Up'] * predict_df['Weighted_Momentum']) / (predict_df['ATR'] + 1e-10)

st.subheader("📋 Integrated Discovery Dashboard")
st.dataframe(
    predict_df[['a_Close', 'Prob_Up', 'Weighted_Momentum', 'Step_Momentum', 'Discovery_Score']].sort_index(ascending=False), 
    use_container_width=True, height=750
)
