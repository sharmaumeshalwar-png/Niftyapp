import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Real Test Matrix", layout="wide")
st.title("🛡️ Nifty 50: Pure Out-of-Sample Signal Matrix Engine")
st.write("A = Nifty Close | B = Kalman Filter (Q=0.0001) | C = Features | D = ML Prediction (No Data Leakage)")
st.write("**Validation Mode:** Train on 2025 Data ➡️ **Strictly Test on 2026 Data (Live Simulation)**")

# -------------------------------------------------------------------------
# Kalman Filter Function
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
# Data Engine
# -------------------------------------------------------------------------
@st.cache_data
def fetch_complete_nifty_data():
    ticker = "^NSEI"
    # 1 Jan 2025 se lekar abhi (June 2026) tak ka data pull
    data = yf.download(ticker, start="2025-01-01", interval="1h")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    if data.empty:
        return pd.DataFrame()
        
    df_res = pd.DataFrame({"Close_A": data['Close'].dropna()})
    return df_res

try:
    with st.spinner("Yahoo Finance se data fetch ho raha hai..."):
        df = fetch_complete_nifty_data()

    if df.empty:
        st.error("Data load nahi ho paya!")
        st.stop()

    # Apply Kalman Filter (B)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)

    # Setup Feature Matrix C
    df['Feature_A_Lag'] = df['Close_A']
    df['Feature_B_Lag'] = df['Kalman_B']
    df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
    
    df_clean = df.dropna().copy()

    # -------------------------------------------------------------------------
    # STRICT TIME-SPLIT (Phase 1 Logic)
    # -------------------------------------------------------------------------
    # 2025 ke end tak ka data sirf training ke liye
    train_mask = df_clean.index < '2026-01-01'
    # 2026 ka poora data test/prediction ke liye (Jise model ne training mein nahi dekha)
    test_mask = df_clean.index >= '2026-01-01'

    df_train = df_clean[train_mask]
    df_test = df_clean[test_mask].copy()

    if df_test.empty:
        st.warning("2026 ka data abhi processed nahi hai, default last 20% data par split lagaya ja raha hai.")
        split_idx = int(len(df_clean) * 0.8)
        df_train = df_clean.iloc[:split_idx]
        df_test = df_clean.iloc[split_idx:].copy()

    # Train Model ONLY on 2025 Data
    X_train = df_train[['Feature_A_Lag', 'Feature_B_Lag']]
    y_train = df_train['Target_D_Next_Hour']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict ONLY on 2026 Data (Pure Blind Test)
    X_test = df_test[['Feature_A_Lag', 'Feature_B_Lag']]
    df_test['ML_Prediction_D'] = model.predict(X_test)

    # -------------------------------------------------------------------------
    # Trading Signal Engine (Based on Blind Test predictions)
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
    # UI Output Matrix Table (Showing ONLY Un-Biased 2026 Results)
    # -------------------------------------------------------------------------
    st.success(f"Model trained on 2025 history. Displaying {len(df_test)} pure blind-test rows from 2026!")
    
    st.markdown("---")
    st.subheader("📋 2026 Asli Trading Signal Matrix Table (No Leakage)")

    # Formatting Table
    output_table = df_test[['Close_A', 'Kalman_B', 'ML_Prediction_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Actual Close (A)", 
        "Kalman Smooth (B)", 
        "ML Next-Hour Predict (D)", 
        "Trading Action Signal"
    ]

    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (Hourly Candle)'}, inplace=True)

    rows_to_show = st.slider("2026 ki pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 50)
    
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Execution Error: {e}")
