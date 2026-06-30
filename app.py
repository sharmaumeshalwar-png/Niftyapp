import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Full 2-Year Matrix", layout="wide")
st.title("🏹 Nifty 50: Complete 1-Hour Signal Matrix (Jan 2025 - Present)")
st.write("A = Nifty Close | B = Kalman Filter (Q=0.0001) | C = Features | D = ML Prediction | 📈 Trading Signals")
st.write("**Data Range:** 01 January 2025 se lekar Aaj tak (Poora Data)")

# -------------------------------------------------------------------------
# Kalman Filter Function (Q = 0.0001 as specified)
# -------------------------------------------------------------------------
def apply_kalman_filter(prices, Q=0.0001, R=0.5):
    n_timestamps = len(prices)
    filtered_prices = np.zeros(n_timestamps)
    if n_timestamps == 0:
        return filtered_prices
        
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

# -------------------------------------------------------------------------
# Data Engine: Fetch and Align Full Data from Jan 2025
# -------------------------------------------------------------------------
@st.cache_data
def fetch_complete_nifty_data():
    ticker = "^NSEI"
    # 1 Jan 2025 se dynamic data fetch
    data = yf.download(ticker, start="2025-01-01", interval="1h")
    
    # Cleaning MultiIndex columns if present in yfinance output
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    if data.empty:
        return pd.DataFrame()
        
    df_res = pd.DataFrame({"Close_A": data['Close'].dropna()})
    return df_res

try:
    with st.spinner("Yahoo Finance se 1 Jan 2025 se aaj tak ka poora Nifty data download ho raha hai..."):
        df = fetch_complete_nifty_data()

    if df.empty:
        st.error("Data load nahi ho paya! Please check internet connection.")
        st.stop()

    # Total rows check
    total_candles_fetched = len(df)

    # -------------------------------------------------------------------------
    # Processing Pipeline (Kalman + ML Feature Synthesis)
    # -------------------------------------------------------------------------
    # Apply Kalman Filter (B)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)

    # Setup Feature Matrix C (Lags)
    df['Feature_A_Lag'] = df['Close_A']
    df['Feature_B_Lag'] = df['Kalman_B']
    
    # Target Variable D: Next Hour's Close
    df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
    df_clean = df.dropna().copy()

    # Train Model on Full Data to predict sequential labels correctly
    X = df_clean[['Feature_A_Lag', 'Feature_B_Lag']]
    y = df_clean['Target_D_Next_Hour']

    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Whole sequence prediction generation (No rows omitted)
    df_clean['ML_Prediction_D'] = model.predict(X)

    # -------------------------------------------------------------------------
    # Rule-Based Trading Signal Engine Generation
    # -------------------------------------------------------------------------
    signals = []
    for idx, row in df_clean.iterrows():
        act_close = row['Close_A']
        kalman_val = row['Kalman_B']
        pred_next = row['ML_Prediction_D']
        
        if (pred_next > act_close) and (act_close > kalman_val):
            signals.append("🟢 BUY")
        elif (pred_next < act_close) and (act_close < kalman_val):
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    df_clean['Trading_Action_Signal'] = signals

    # -------------------------------------------------------------------------
    # UI Output Matrix Display (Poora Data Show Hoga)
    # -------------------------------------------------------------------------
    st.success(f"Successfully loaded {total_candles_fetched} hourly candles from 1 Jan 2025!")
    
    st.markdown("---")
    st.subheader(f"📋 Full Nifty Matrix Table ({len(df_clean)} Clean Records)")

    # Formatting Table
    output_table = df_clean[['Close_A', 'Kalman_B', 'ML_Prediction_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Actual Close (A)", 
        "Kalman Smooth (B)", 
        "ML Next-Hour Predict (D)", 
        "Trading Action Signal"
    ]

    # Convert Index to Date Time Strings
    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (Hourly Candle)'}, inplace=True)

    # Dynamic Slider to change views (Max range up to full data size)
    rows_to_show = st.slider("Table mein pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 100)
    
    # Render full table without cutting data
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Execution Error: {e}")
