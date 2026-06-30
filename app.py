import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Daily Strategy Matrix", layout="wide")
st.title("🏹 Nifty 50: Pure Daily (1-Day) Signal Matrix Engine")
st.write("A = Nifty Daily Close | B = Kalman Filter (Q=0.0001) | C = Features | D = ML Prediction")
st.write("**Train Window:** 01 July 2024 ➡️ 01 July 2025 (Daily) | **Test Window:** 01 July 2025 ➡️ Aaj Tak")

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
# Data Engine: 1-Day Candles from 1 July 2024 to Present
# -------------------------------------------------------------------------
@st.cache_data
def fetch_daily_nifty_data():
    ticker = "^NSEI"
    # 1 July 2024 se aaj tak ka Daily (1d) data fetch karna
    data = yf.download(ticker, start="2024-07-01", end="2026-07-01", interval="1d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    if data.empty:
        return pd.DataFrame()
        
    df_res = pd.DataFrame({"Close_A": data['Close'].dropna()})
    return df_res

try:
    with st.spinner("Yahoo Finance se Nifty ka 1-Day data download ho raha hai..."):
        df = fetch_daily_nifty_data()

    if df.empty:
        st.error("Data load nahi ho paya! Internet check karein.")
        st.stop()

    # Apply Kalman Filter (B)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)

    # Setup Feature Matrix C
    df['Feature_A_Lag'] = df['Close_A']
    df['Feature_B_Lag'] = df['Kalman_B']
    df['Target_D_Next_Day'] = df['Close_A'].shift(-1)
    
    df_clean = df.dropna().copy()

    # -------------------------------------------------------------------------
    # STRICT TIMEFRAME SPLIT (User Instructions)
    # -------------------------------------------------------------------------
    # 1 July 2024 se 1 July 2025 tak sirf Training
    train_mask = (df_clean.index >= '2024-07-01') & (df_clean.index < '2025-07-01')
    # 1 July 2025 se Aaj tak (June 2026) sirf Testing / Predictions
    test_mask = df_clean.index >= '2025-07-01'

    df_train = df_clean[train_mask]
    df_test = df_clean[test_mask].copy()

    if df_train.empty or df_test.empty:
        st.error("Time boundaries ke hisab se data split fail ho gaya. Data check karein.")
        st.stop()

    # Train Model strictly on 2024-2025 Daily data
    X_train = df_train[['Feature_A_Lag', 'Feature_B_Lag']]
    y_train = df_train['Target_D_Next_Day']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict blindly on 2025-2026 Daily data
    X_test = df_test[['Feature_A_Lag', 'Feature_B_Lag']]
    df_test['ML_Prediction_D'] = model.predict(X_test)

    # -------------------------------------------------------------------------
    # Trading Signal Engine (1-Day Candle Rules)
    # -------------------------------------------------------------------------
    signals = []
    for idx, row in df_test.iterrows():
        act_close = row['Close_A']
        kalman_val = row['Kalman_B']
        pred_next = row['ML_Prediction_D']
        
        if (pred_next > act_close) and (act_close > kalman_val):
            signals.append("🟢 BUY")
        elif (pred_next < act_close) and (act_close < kalman_val):
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    df_test['Trading_Action_Signal'] = signals

    # -------------------------------------------------------------------------
    # UI Output Matrix Table (Showing 1 July 2025 to Present Results)
    # -------------------------------------------------------------------------
    st.success(f"Model trained on 1 Year Daily Data. Displaying {len(df_test)} Clean Daily Predictions from 01 July 2025 onwards!")
    
    st.markdown("---")
    st.subheader("📋 1-Day Interval Signal Matrix Table (Un-Biased Actual Test)")

    # Table Formatting
    output_table = df_test[['Close_A', 'Kalman_B', 'ML_Prediction_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Daily Actual Close (A)", 
        "Kalman Smooth (B)", 
        "ML Next-Day Predict (D)", 
        "Trading Action Signal"
    ]

    output_table.index = output_table.index.strftime('%Y-%m-%d')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date (1-Day Candle)'}, inplace=True)

    # Dynamic Slider to view rows
    rows_to_show = st.slider("Pichle kitne trading dino ka data dekhna hai?", 10, len(output_table), 50)
    
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Execution Error: {e}")
