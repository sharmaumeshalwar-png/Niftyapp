import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine 2.0", layout="wide")
st.title("📊 Nifty 50 [2-Year] Hybrid Double Kalman Engine [50:50 Split]")

# =====================================================================
# MATHEMATICAL ENGINE: FILTERS
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0, q=0.0001, r=2.5): 
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p  
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
    x, p, q, r = data_array[0], 1.0, 0.05, 0.2
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# DATA PIPELINE
# =====================================================================
with st.spinner("Downloading 2-Year Nifty 1-Hour Data & Processing Matrices..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    if raw_df.empty:
        st.error("Market data unavailable.")
        st.stop()
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(inplace=True)

    # Base Matrix
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Imbalance'] = ((df['Open'] + df['a_Close'])/2 - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(1), 1, 0)
    df_clean = df.dropna(subset=features_matrix + ['Target']).copy()

# =====================================================================
# DYNAMIC SPLIT ENGINE (50:50)
# =====================================================================
split_idx = int(len(df_clean) * 0.50)
df_train = df_clean.iloc[:split_idx]
df_predict = df_clean.iloc[split_idx:].copy()

# Model Fitting
model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model_flow.fit(df_train[features_matrix], df_train['Target'])
probs = model_flow.predict_proba(df_predict[features_matrix])
df_predict['Prob_Up'] = probs[:, 1]
df_predict['Prob_Down'] = probs[:, 0]

# =====================================================================
# LIVE TREND-LOCK CIRCUIT
# =====================================================================
final_signals, accumulator = [], 0
for i in range(len(df_predict)):
    p_up, p_down = df_predict['Prob_Up'].iloc[i], df_predict['Prob_Down'].iloc[i]
    if p_up >= 0.55: accumulator += 1
    elif p_down >= 0.55: accumulator -= 1
    accumulator = max(-5, min(5, accumulator))
    
    if accumulator == 5: final_signals.append("🟢 STRONG BUY")
    elif accumulator == -5: final_signals.append("🔴 STRONG SELL")
    else: final_signals.append(f"⚪ NEUTRAL (Score: {accumulator})")

df_predict['d_ML_Signal'] = final_signals
df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['c_Combined'].values, initial_p=0.50)

# Display
display_df = df_predict[['a_Close', 'Prob_Up', 'Weighted_Momentum', 'd_ML_Signal']].sort_index(ascending=False)
st.subheader("📋 Engine Dashboard")
st.dataframe(display_df.head(50), use_container_width=True)
