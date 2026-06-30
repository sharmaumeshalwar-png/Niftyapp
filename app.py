import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty 1-Hour Error Matrix", layout="wide")
st.title("🏹 Nifty 50: 1-Hour Pure Signal Matrix Engine")
st.write("A = Nifty Close | B = Kalman Filter (Q=0.0001) | D = ML Prediction | **A - D = Prediction Difference**")
st.write("**Train Window:** 01 July 2024 ➡️ 01 July 2025 (Hourly) | **Test Window:** 01 July 2025 ➡️ Aaj Tak")

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
# Data Engine: 1-Hour Candles from 1 July 2024 to Present
# -------------------------------------------------------------------------
@st.cache_data
def fetch_hourly_nifty_data():
    ticker = "^NSEI"
    data = yf.download(ticker, start="2024-07-01", interval="1h")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    if data.empty:
        return pd.DataFrame()
        
    df_res = pd.DataFrame({"Close_A": data['Close'].dropna()})
    return df_res

try:
    with st.spinner("Yahoo Finance se Nifty ka 1-Hour data download ho raha hai..."):
        df = fetch_hourly_nifty_data()

    if df.empty:
        st.error("Data load nahi ho paya! Internet check karein.")
        st.stop()

    # Apply Kalman Filter (B)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)

    # Setup Feature Matrix C
    df['Feature_A_Lag'] = df['Close_A']
    df['Feature_B_Lag'] = df['Kalman_B']
    df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
    
    df_clean = df.dropna().copy()

    # -------------------------------------------------------------------------
    # STRICT TIMEFRAME SPLIT (User Instructions)
    # -------------------------------------------------------------------------
    train_mask = (df_clean.index >= '2024-07-01') & (df_clean.index < '2025-07-01')
    test_mask = df_clean.index >= '2025-07-01'

    df_train = df_clean[train_mask]
    df_test = df_clean[test_mask].copy()

    if df_train.empty or df_test.empty:
        st.error("Time boundaries ke hisab se data split fail ho gaya. Yahoo Finance par pichla 1-hour data limited ho sakta hai.")
        st.stop()

    # Train Model strictly on 2024-2025 Hourly history
    X_train = df_train[['Feature_A_Lag', 'Feature_B_Lag']]
    y_train = df_train['Target_D_Next_Hour']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict blindly on 2025-2026 1-Hour data
    X_test = df_test[['Feature_A_Lag', 'Feature_B_Lag']]
    df_test['ML_Prediction_D'] = model.predict(X_test)

    # -------------------------------------------------------------------------
    # Calculation: A - D (Actual Close - ML Prediction)
    # -------------------------------------------------------------------------
    df_test['Diff_A_minus_D'] = df_test['Close_A'] - df_test['ML_Prediction_D']

    # -------------------------------------------------------------------------
    # Trading Signal Engine (1-Hour Candle Rules)
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
    # UI Output Matrix Table with A-D Column
    # -------------------------------------------------------------------------
    st.success(f"Model successfully trained. Displaying {len(df_test)} Clean Candles from 01 July 2025 onwards!")
    
    st.markdown("---")
    st.subheader("📋 1-Hour Interval Signal & Error Matrix Table")

    # Table Formatting (Added Difference Column)
    output_table = df_test[['Close_A', 'Kalman_B', 'ML_Prediction_D', 'Diff_A_minus_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Actual Close (A)", 
        "Kalman Smooth (B)", 
        "ML Next-Hour Predict (D)", 
        "Difference (A - D)",
        "Trading Action Signal"
    ]

    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (1-Hour Candle)'}, inplace=True)

    # Dynamic Slider to view rows
    rows_to_show = st.slider("Pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 50)
    
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Execution Error: {e}")
