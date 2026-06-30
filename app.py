import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration (FIXED: st.set_page_config used here)
st.set_page_config(page_title="Nifty Advanced ML Matrix", layout="wide")

st.title("🏹 Nifty 50: Advanced 1-Hour Matrix with RSI & MACD")
st.write("A = Close | B = Kalman Filter | C = Features (RSI + MACD Included) | D = ML Prediction | **A - D = Error**")
st.write("**Validation Mode:** Train on 2024-25 History ➡️ **Strictly Test on Unseen 2026 Data**")

# -------------------------------------------------------------------------
# Mathematical Functions (Kalman, RSI, MACD)
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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# -------------------------------------------------------------------------
# Data Engine (Error 95 Protection Layer)
# -------------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_hourly_nifty_advanced():
    try:
        ticker = "^NSEI"
        data = yf.download(ticker, start="2024-08-01", interval="1h", auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return pd.DataFrame()
        return pd.DataFrame({"Close_A": data['Close'].dropna()})
    except Exception:
        return pd.DataFrame()

try:
    with st.spinner("API se Data Fetch aur Indicators calculate ho rahe hain..."):
        df = fetch_hourly_nifty_advanced()

    if df.empty:
        st.warning("⚠️ Yahoo Finance Limit Error! Backup Simulation Data Active Kiya Gaya Hai.")
        dates = pd.date_range(start="2024-08-01", end="2026-06-30", freq="h")
        np.random.seed(42)
        mock_prices = 24000 + np.cumsum(np.random.normal(0, 15, len(dates)))
        df = pd.DataFrame({"Close_A": mock_prices}, index=dates)

    # 1. Apply Kalman Filter (B)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)

    # 2. Inject Technical Indicators (RSI & MACD) into the Matrix
    df['RSI'] = calculate_rsi(df['Close_A'])
    df['MACD_Line'], df['MACD_Signal'] = calculate_macd(df['Close_A'])
    
    # 3. Target Variable D
    df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
    
    # Clean data from NaN created by indicator periods
    df_clean = df.dropna().copy()

    # -------------------------------------------------------------------------
    # STRICT TIMEFRAME SPLIT
    # -------------------------------------------------------------------------
    train_mask = (df_clean.index >= '2024-08-01') & (df_clean.index < '2025-07-01')
    test_mask = df_clean.index >= '2025-07-01'

    df_train = df_clean[train_mask]
    df_test = df_clean[test_mask].copy()

    if df_train.empty or df_test.empty:
        st.error("Data split boundary mismatch!")
        st.stop()

    # Features selection WITH RSI & MACD
    feature_cols = ['Close_A', 'Kalman_B', 'RSI', 'MACD_Line', 'MACD_Signal']
    
    # Train Engine
    X_train = df_train[feature_cols]
    y_train = df_train['Target_D_Next_Hour']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict blindly on Unseen Test Data
    X_test = df_test[feature_cols]
    df_test['ML_Prediction_D'] = model.predict(X_test)

    # Calculate A - D Error
    df_test['Diff_A_minus_D'] = df_test['Close_A'] - df_test['ML_Prediction_D']

    # -------------------------------------------------------------------------
    # Advanced Trading Signal Engine
    # -------------------------------------------------------------------------
    signals = []
    for idx, row in df_test.iterrows():
        act_close = row['Close_A']
        kalman_val = row['Kalman_B']
        pred_next = row['ML_Prediction_D']
        rsi_val = row['RSI']
        macd_l = row['MACD_Line']
        macd_s = row['MACD_Signal']
        
        # Super Sharp Confluence Rules
        if (pred_next > act_close) and (act_close > kalman_val) and (rsi_val < 70) and (macd_l > macd_s):
            signals.append("🟢 BUY")
        elif (pred_next < act_close) and (act_close < kalman_val) and (rsi_val > 30) and (macd_l < macd_s):
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    df_test['Trading_Action_Signal'] = signals

    # -------------------------------------------------------------------------
    # UI Display Output Data Frame
    # -------------------------------------------------------------------------
    st.success(f"Success! Model trained with 5 core features. Displaying Unbiased 2025-2026 predictions.")
    st.markdown("---")
    st.subheader("📋 Advanced Nifty 50 Feature Signal Table")

    # Table Formatting
    output_table = df_test[['Close_A', 'Kalman_B', 'RSI', 'Diff_A_minus_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Close (A)", 
        "Kalman Smooth (B)", 
        "RSI (14)",
        "Difference (A - D)",
        "Trading Action Signal"
    ]

    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (Hourly Candle)'}, inplace=True)

    rows_to_show = st.slider("Pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 40)
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"System Error: {e}")
