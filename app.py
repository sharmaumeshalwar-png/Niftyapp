import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Linear Smooth Distance Kalman + Non-Linear Step Momentum")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR FILTER
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0):
    if len(data_array) == 0:
        return []
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
# MATHEMATICAL ENGINE 2: NON-LINEAR FILTER
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0:
        return []
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

# Data Loading
with st.spinner("Aligning 25-Candle Dual Kalman Matrices..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    if raw_df.empty:
        st.error("Data error. Please check internet or API.")
        st.stop()

    df = raw_df.copy()
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df.ffill()

    # Engine Logic
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Feature Engineering
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    df.dropna(inplace=True)

    # Simple Model
    features = ['c_Combined', 'Order_Imbalance', 'Normalized_Gap']
    model = RandomForestClassifier(n_estimators=100).fit(df[features].iloc[:-1], df['Target'].iloc[:-1])
    
    st.write("Engine is ready and running!")
    st.dataframe(df.tail(10))
