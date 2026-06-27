import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty Dual Kalman Filter (Fast vs Slow)")

st.write("1 Jan 2025 se 1-Hour Frozen Data par Dual Kalman chal raha hai...")

# 1. Data Download
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

if data.empty:
    st.error("Yahoo Finance data nahi de raha hai. Interval='1d' check karein.")
else:
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    a = data['Close'].values.flatten()
    num_steps = len(a)

    # 2. FAST KALMAN FILTER SETUP
    b_fast = np.zeros(num_steps)
    x_fast = a[0]
    P_f, Q_f, R_f = 1.0, 0.5, 1.0  # Fast tuning
    for t in range(num_steps):
        K = (P_f + Q_f) / (P_f + Q_f + R_f)
        x_fast = x_fast + K * (a[t] - x_fast)
        P_f = (1 - K) * (P_f + Q_f)
        b_fast[t] = x_fast

    # 3. SLOW KALMAN FILTER SETUP
    b_slow = np.zeros(num_steps)
    x_slow = a[0]
    P_s, Q_s, R_s = 1.0, 0.01, 10.0  # Slow tuning
    for t in range(num_steps):
        K = (P_s + Q_s) / (P_s + Q_s + R_s)
        x_slow = x_slow + K * (a[t] - x_slow)
        P_s = (1 - K) * (P_s + Q_s)
        b_slow[t] = x_slow

    # 4. Calculate Difference (Hint Generator)
    c_diff = b_fast - b_slow

    # 5. Create Automatic Signals
    signals = []
    for val in c_diff:
        if val > 5:
            signals.append("🟢 BUY TREND")
        elif val < -5:
            signals.append("🔴 SELL TREND")
        else:
            signals.append("⚪ SIDEWAYS / NO TRADE")

    # 6. Creating the Table
    df_table = pd.DataFrame({
        'a: Nifty Close': np.round(a, 2),
        'b_fast (Short-term)': np.round(b_fast, 2),
        'b_slow (Long-term)': np.round(b_slow, 2),
        'c_diff (Fast - Slow)': np.round(c_diff, 2),
        'System Signal': signals
    }, index=timestamps)

    # 7. Reverse the table to show latest data on top
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    
    st.success("Dual Kalman Filter successfully deployed!")
