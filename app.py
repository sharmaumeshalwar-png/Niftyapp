import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty Fixed Multiplier (10 & 50) Anti-Gap Filter")

st.write("Fast Q=10.0 Aur Slow R=50.0 Par Math Fixed Hai...")

# 1. Data Download
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

if data.empty:
    st.error("Data nahi mil pa raha hai.")
else:
    data['Date'] = data.index.date
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    a = data['Close'].values.flatten()
    o = data['Open'].values.flatten()
    num_steps = len(a)

    # 2. GAP CUMULATIVE OFFSETS
    a_adjusted = np.copy(a)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        current_date = data['Date'].iloc[t]
        prev_date = data['Date'].iloc[t-1]
        
        if current_date != prev_date:
            gap = o[t] - a[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        
        a_adjusted[t] = a[t] - cumulative_gap

    # 3. FAST KALMAN (Fixed Q = 10.0)
    b_fast = np.zeros(num_steps)
    x_fast = a_adjusted[0]
    P_f, Q_f, R_f = 1.0, 10.0, 1.0
    for t in range(num_steps):
        K = (P_f + Q_f) / (P_f + Q_f + R_f)
        x_fast = x_fast + K * (a_adjusted[t] - x_fast)
        P_f = (1 - K) * (P_f + Q_f)
        b_fast[t] = x_fast

    # 4. SLOW KALMAN (Fixed R = 50.0)
    b_slow = np.zeros(num_steps)
    x_slow = a_adjusted[0]
    P_s, Q_s, R_s = 1.0, 0.01, 50.0
    for t in range(num_steps):
        K = (P_s + Q_s) / (P_s + Q_s + R_s)
        x_slow = x_slow + K * (a_adjusted[t] - x_slow)
        P_s = (1 - K) * (P_s + Q_s)
        b_slow[t] = x_slow

    # 5. Re-Scaling
    b_fast_real = b_fast + cumulative_gap
    b_slow_real = b_slow + cumulative_gap
    c_diff = b_fast_real - b_slow_real

    # 6. FIXED ANTI-FLIP HYSTERESIS (Barrier = 20)
    signals = []
    current_state = "⚪ SIDEWAYS"
    barrier_gate = 20.0

    for val in c_diff:
        if current_state == "⚪ SIDEWAYS":
            if val > barrier_gate:
                current_state = "🟢 BUY TREND"
            elif val < -barrier_gate:
                current_state = "🔴 SELL TREND"
        elif current_state == "🟢 BUY TREND":
            if val < -barrier_gate:
                current_state = "🔴 SELL TREND"
        elif current_state == "🔴 SELL TREND":
            if val > barrier_gate:
                current_state = "🟢 BUY TREND"
        signals.append(current_state)

    # 7. Creating Table Data
    df_table = pd.DataFrame({
        'a: Nifty Close': np.round(a, 2),
        'b_fast (Q=10)': np.round(b_fast_real, 2),
        'b_slow (R=50)': np.round(b_slow_real, 2),
        'c_diff (Spread)': np.round(c_diff, 2),
        'Final Stable Signal': signals
    }, index=timestamps)

    # 8. Render Table (Latest Top)
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("System completely optimized for Q=10 and R=50!")
