import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")

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
# DATA ENGINE
# =====================================================================
with st.spinner("Processing 2-Year Nifty Data..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if raw_df.empty:
        st.error("Data load failed. Check internet.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    df['Close'] = raw_df['Close'].ffill()
    df['Open'] = raw_df['Open'].ffill()
    df['High'] = raw_df['High'].ffill()
    df['Low'] = raw_df['Low'].ffill()
    df.dropna(inplace=True)

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

# =====================================================================
# 50:50 SPLIT ENGINE
# =====================================================================
split_idx = int(len(df) * 0.50)
train_df = df.iloc[:split_idx]
predict_df = df.iloc[split_idx:].copy()

features = ['c_Combined', 'Order_Imbalance', 'Normalized_Gap']
model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(train_df[features], train_df['Target'])

# Predictions
probs = model.predict_proba(predict_df[features])
predict_df['Prob_Down'] = probs[:, 0]
predict_df['Prob_Up'] = probs[:, 1]

# Kalman Momentum (0.50 parameter as requested)
predict_df['Weighted_Momentum'] = apply_kalman_filter_custom(predict_df['c_Combined'].values, initial_p=0.50)
predict_df['Step_Momentum'] = np.round(apply_non_linear_kalman_momentum(predict_df['Weighted_Momentum'].values))

# Display results
st.subheader("📋 Dashboard (Full View)")
st.dataframe(
    predict_df[['a_Close', 'Prob_Up', 'Prob_Down', 'Weighted_Momentum', 'Step_Momentum']].sort_index(ascending=False), 
    use_container_width=True, 
    height=750
)
