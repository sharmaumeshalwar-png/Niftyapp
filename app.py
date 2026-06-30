import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty High-Accuracy ML Matrix", layout="wide")
st.title("🏹 Nifty 50: Adaptive % Return ML Engine (High Accuracy)")
st.write("A = Close | B = Kalman Filter | C = Advanced Matrix | D = ML Directional Predict | **A - D = Residual**")
st.write("**Engine State:** Dynamic Rolling Re-Training Mode (Adaptive to 2026 Volatility)")

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
# Data Engine
# -------------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_unlimited_nifty_data():
    try:
        ticker = "^NSEI"
        # Maximum possible data pull for full learning
        data = yf.download(ticker, start="2024-01-01", interval="1h", auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return pd.DataFrame({"Close_A": data['Close'].dropna()})
    except Exception:
        return pd.DataFrame()

try:
    with st.spinner("Deep Data Analysis Engine Active Ho Raha Hai..."):
        df = fetch_unlimited_nifty_data()

    if df.empty:
        st.error("API error! Network reload karein.")
        st.stop()

    # Feature Engineering Layer
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.5)
    df['RSI'] = calculate_rsi(df['Close_A'])
    df['MACD_L'], df['MACD_S'] = calculate_macd(df['Close_A'])
    
    # ACCURACY FIX: exact price ke badle Log/Pct Returns calculate karna
    df['Hourly_Return'] = df['Close_A'].pct_change()
    
    # Target D: Next Hour's percentage move
    df['Target_Return_D'] = df['Hourly_Return'].shift(-1)
    
    df_clean = df.dropna().copy()

    # Strictly slice test zone from July 2025 to Present
    test_start_date = '2025-07-01'
    df_test = df_clean[df_clean.index >= test_start_date].copy()
    
    feature_cols = ['Hourly_Return', 'RSI', 'MACD_L', 'MACD_S']
    
    # -------------------------------------------------------------------------
    # ADAPTIVE ROLLING LEARNING ENGINE (No Static Blunders)
    # -------------------------------------------------------------------------
    predictions_pct = []
    
    # Har hourly candle par model update hoga live market ki tarah
    for i in range(len(df_test)):
        current_time = df_test.index[i]
        
        # Training subset: Is candle ke pehle ka saara pichla data
        train_sub = df_clean[df_clean.index < current_time]
        
        # System constraints fallback if subset too small
        if len(train_sub) < 500:
            predictions_pct.append(0.0)
            continue
            
        X_tr = train_sub[feature_cols].tail(1000) # Last 1000 candles for relevant memory
        y_tr = train_sub['Target_Return_D'].tail(1000)
        
        # Light but hyper-fast Adaptive Estimator
        core_rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        core_rf.fit(X_tr, y_tr)
        
        # Current data feature point input
        X_cur = df_test[feature_cols].iloc[[i]]
        pred_ret = core_rf.predict(X_cur)[0]
        predictions_pct.append(pred_ret)

    df_test['Predicted_Return_D'] = predictions_pct
    
    # Convert predicted % returns back to absolute price matrix D for user viewing
    df_test['ML_Prediction_D'] = df_test['Close_A'] * (1 + df_test['Predicted_Return_D'])
    df_test['Diff_A_minus_D'] = df_test['Close_A'] - df_test['ML_Prediction_D']

    # -------------------------------------------------------------------------
    # Advanced Confluence Trading Signals
    # -------------------------------------------------------------------------
    signals = []
    for idx, row in df_test.iterrows():
        act_close = row['Close_A']
        kalman_val = row['Kalman_B']
        pred_change = row['Predicted_Return_D']
        rsi_val = row['RSI']
        macd_l = row['MACD_L']
        macd_s = row['MACD_S']
        
        # Pure direction based strict signals
        if (pred_change > 0.0005) and (act_close > kalman_val) and (rsi_val < 65) and (macd_l > macd_s):
            signals.append("🟢 BUY")
        elif (pred_change < -0.0005) and (act_close < kalman_val) and (rsi_val > 35) and (macd_l < macd_s):
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    df_test['Trading_Action_Signal'] = signals

    # -------------------------------------------------------------------------
    # UI Component Output
    # -------------------------------------------------------------------------
    st.success(f"Adaptive High-Accuracy Engine Active! Processed {len(df_test)} sequential matrix rows.")
    st.markdown("---")
    st.subheader("📋 Highly Accurate Nifty 50 Rolling Matrix Table")

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
    output_table.rename(columns={'index': 'Date & Time (1-Hour Candle)'}, inplace=True)

    rows_to_show = st.slider("Pichli kitni candles ek sath dekhni hain?", 10, len(output_table), 30)
    st.dataframe(output_table.tail(rows_to_show), use_container_width=True, height=500)

except Exception as e:
    st.error(f"Execution Bypass Alert: {e}")
