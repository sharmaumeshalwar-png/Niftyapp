import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Discovery Engine", layout="wide")
st.title("📊 Nifty 50 Discovery Engine [Volatility-Aware 0.50 Engine]")

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

# =====================================================================
# DATA & DISCOVERY ENGINE
# =====================================================================
with st.spinner("Analyzing Volatility & Nifty Structure..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if raw_df.empty:
        st.error("Data load failed.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    df['Close'] = raw_df['Close'].ffill()
    df['High'] = raw_df['High'].ffill()
    df['Low'] = raw_df['Low'].ffill()
    df.dropna(inplace=True)

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # POINT 1: VOLATILITY FEATURE ADDED
    df['Volatility'] = df['a_Close'].rolling(window=24).std()
    
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['Volatility'] + 1e-10)
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# MODEL TRAINING WITH NEW FEATURES
# =====================================================================
split_idx = int(len(df) * 0.50)
train_df = df.iloc[:split_idx]
predict_df = df.iloc[split_idx:].copy()

# New feature set including Volatility
features = ['c_Combined', 'Order_Imbalance', 'Normalized_Gap', 'Volatility']
model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(train_df[features], train_df['Target'])

# Predictions
probs = model.predict_proba(predict_df[features])
predict_df['Prob_Up'] = probs[:, 1]

# Display Results
st.subheader("📋 Discovery Engine (Volatility-Aware)")
st.dataframe(
    predict_df[['a_Close', 'Volatility', 'Prob_Up']].sort_index(ascending=False), 
    use_container_width=True, 
    height=750
)
