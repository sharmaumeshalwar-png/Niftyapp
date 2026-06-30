import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty July 2025 Timeline Engine", layout="wide")
st.title("🏹 Nifty 50: Historical ML Engine (From 1 July 2025)")
st.write("A = Nifty Actual Close | B = Kalman Filter | C = Features | D = ML Predict | **A - D = Residual**")
st.write("**Engine Range Status:** Strictly Analyzing July 2025 to June 2026 Window")

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
# Dynamic Deep Data Stream Engine
# -------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_nifty_july_timeline():
    ticker = "^NSEI"
    try:
        session = yf.utils.get_ticker_anonymous_session()
        data = yf.download(
            ticker, 
            start="2024-06-01", 
            interval="1h", 
            auto_adjust=True,
            session=session
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if not data.empty and len(data) > 100:
            df_res = pd.DataFrame({"Close_A": data['Close'].dropna()})
            if df_res["Close_A"].iloc[-1] > 10000:
                return df_res
    except Exception:
        pass
        
    return pd.DataFrame()

try:
    with st.spinner("July 2025 Block Matrix Download Ho Raha Hai..."):
        df = fetch_nifty_july_timeline()

    if df.empty:
        st.error("❌ NSE Data core access busy! Kripya reload karein.")
        st.stop()

    # Feature Engineering Layer
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)
    df['RSI'] = calculate_rsi(df['Close_A'])
    df['MACD_L'], df['MACD_S'] = calculate_macd(df['Close_A'])
    
    df['Hourly_Return'] = df['Close_A'].pct_change()
    df['Target_Return_D'] = df['Hourly_Return'].shift(-1)
    
    df_clean = df.dropna().copy()

    # Timeline filter setup
    timeline_start = '2025-07-01'
    df_test = df_clean[df_clean.index >= timeline_start].copy()
    
    if df_test.empty:
        df_test = df_clean.tail(150).copy()
        
    feature_cols = ['Hourly_Return', 'RSI', 'MACD_L', 'MACD_S']
    predictions_pct = []
    
    df_test_subset = df_test.tail(200).copy() 
    
    # -------------------------------------------------------------------------
    # Sequential Window Learning (July 2025 Forward)
    # -------------------------------------------------------------------------
    for i in range(len(df_test_subset)):
        current_time = df_test_subset.index[i]
        train_sub = df_clean[df_clean.index < current_time]
        
        X_tr = train_sub[feature_cols].tail(400)
        y_tr = train_sub['Target_Return_D'].tail(400)
        
        core_rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        core_rf.fit(X_tr, y_tr)
        
        X_cur = df_test_subset[feature_cols].iloc[[i]]
        pred_ret = core_rf.predict(X_cur)[0]
        predictions_pct.append(pred_ret)

    df_test_subset['Predicted_Return_D'] = predictions_pct
    df_test_subset['ML_Prediction_D'] = df_test_subset['Close_A'] * (1 + df_test_subset['Predicted_Return_D'])
    df_test_subset['Diff_A_minus_D'] = df_test_subset['Close_A'] - df_test_subset['ML_Prediction_D']

    # -------------------------------------------------------------------------
    # Trading Signal Engine (FIXED INDENTATION BLOCK START)
    # -------------------------------------------------------------------------
    signals = []
    for idx, row in df_test_subset.iterrows():
        act_close = row['Close_A']
        kalman_val = row['Kalman_B']
        pred_change = row['Predicted_Return_D']
        rsi_val = row['RSI']
        macd_l = row['MACD_L']
        macd_s = row['MACD_S']
        
        if (pred_change > 0.0001) and (act_close > kalman_val) and (rsi_val < 68) and (macd_l > macd_s):
            signals.append("🟢 BUY")
        elif (pred_change < -0.0001) and (act_close < kalman_val) and (rsi_val > 32) and (macd_l < macd_s):
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    df_test_subset['Trading_Action_Signal'] = signals

    # -------------------------------------------------------------------------
    # UI Display Generation
    # -------------------------------------------------------------------------
    st.success("🎯 Setup Completed Successfully! Timelines recalculated from July 2025.")
    st.markdown("---")
    
    output_table = df_test_subset[['Close_A', 'Kalman_B', 'RSI', 'Diff_A_minus_D', 'Trading_Action_Signal']].copy()
    output_table.columns = [
        "Nifty Close (A)", 
        "Kalman Smooth (B)", 
        "RSI (14)",
        "Difference (A - D)",
        "Trading Action Signal"
    ]

    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (1-Hour Candle)'}, inplace=True)

    rows_to_show = st.slider("Pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 40)
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Fatal Interrupt Bypass Active: {e}")
