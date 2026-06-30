import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Adaptive Rolling Matrix", layout="wide")
st.title("🚀 Nifty 50: Adaptive Rolling Matrix Engine (1-Hour)")
st.write("A = Close | B = Kalman (Q=0.0001) | D = ML Rolling Predict | **A - D = Dynamic Variance**")
st.write("**Strategy:** 🔁 Every hour, the model re-learns the last 200 hours to predict the next hour.")

# -------------------------------------------------------------------------
# Technical Indicators & Kalman Functions
# -------------------------------------------------------------------------
def apply_kalman_filter(prices, Q=0.0001, R=0.5):
    n_timestamps = len(prices)
    filtered_prices = np.zeros(n_timestamps)
    if n_timestamps == 0: return filtered_prices
    x_hat = prices[0]  
    P = 1.0            
    for t in range(n_timestamps):
        x_hat_minus = x_hat
        P_minus = P + Q
        K = P_minus / (P_minus + R)  
        x_hat = x_hat_minus + K * (prices[t] - x_hat_minus)
        P = (1 - K) * P_minus
        filtered_prices[t] = x_hat
    return filtered_prices

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / (down + 1e-10)
    rsi = np.zeros_like(prices)
    rsi[:period] = 50.0
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / (down + 1e-10)
        rsi[i] = 100. - (100. / (1. + rs))
    return rsi

# -------------------------------------------------------------------------
# Data Pulling Engine
# -------------------------------------------------------------------------
@st.cache_data
def fetch_nifty_data():
    ticker = "^NSEI"
    # Pure 2025 se 2026 tak ka continuous structural data load
    data = yf.download(ticker, start="2025-01-01", interval="1h")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty: return pd.DataFrame()
    return pd.DataFrame({"Close_A": data['Close'].dropna()})

try:
    with st.spinner("Nifty ka data live engine par process ho raha hai..."):
        df = fetch_nifty_data()

    if df.empty:
        st.error("Data load nahi ho paya!")
        st.stop()

    # Feature Engineering
    prices = df['Close_A'].values
    df['Kalman_B'] = apply_kalman_filter(prices, Q=0.0001, R=0.5)
    df['RSI'] = calculate_rsi(prices, period=14)
    df['Price_Vel'] = df['Close_A'].diff(3).fillna(0) # Pichle 3 ghante ka momentum
    
    # Target Shifting
    df['Target_D'] = df['Close_A'].shift(-1)
    df_clean = df.dropna().copy()

    # -------------------------------------------------------------------------
    # WALKING / ROLLING RETRAINING ENGINE (The Real Fix)
    # -------------------------------------------------------------------------
    # Hum sirf 2026 ke data par predictions show karenge, par model 2025 se seekhna shuru karega
    test_start_date = '2026-01-01'
    df_test_zone = df_clean[df_clean.index >= test_start_date].copy()
    
    ml_predictions = []
    window_size = 200 # Har baar pichle 200 ghante ka data padhega

    with st.spinner("Model live re-training loop chala raha hai... (Is
