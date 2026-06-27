import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty Multiplier-Tuned No-Flip Kalman Filter")

# ---- TUNING PANEL (YAHAN SE MULTIPLIER CHANGE KRO) ----
st.sidebar.header("Kalman Multiplier Settings")
fast_multiplier = st.sidebar.slider("Fast Kalman Sensitivity (Q)", 0.1, 2.0, 1.2, step=0.1)
slow_multiplier = st.sidebar.slider("Slow Kalman Smoothness (R)", 5.0, 50.0, 25.0, step=1.0)
barrier_gate = st.sidebar.slider("Anti-Flip Barrier Gate", 5.0, 50.0, 20.0, step=1.0)

st.write(f"Current Settings -> Fast Q: {fast_multiplier} | Slow R: {slow_multiplier} | Barrier: {barrier_gate}")

# 1. Data Download
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

if data.empty:
    st.error("Data nahi mil pa raha hai.")
else:
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    a = data['Close'].values.flatten()
    num_steps = len(a)

    # 2. FAST KALMAN (Using tuned fast_multiplier)
    b_fast = np.zeros(num_steps)
    x_fast = a[0]
    P_f = 1.0
    Q_f = fast_multiplier  # Higher means more sensitive to price jumps
    R_f = 1.0
    for t in range(num_steps):
        K = (P_f + Q_f) / (P_f + Q_f + R_f)
        x_fast = x_fast + K * (a[t] - x_fast)
        P_f = (1 - K) * (P_f + Q_f)
        b_fast[t] = x_fast

    # 3. SLOW KALMAN (Using tuned slow_multiplier)
    b_slow = np.zeros(num_steps)
    x_slow = a[0]
    P_s = 1.0
    Q_s = 0.01
    R_s = slow_multiplier  # Higher means ignores short-term noise completely
    for t in range(num_steps):
        K = (P_s + Q_s) / (P_s + Q_s + R_s)
        x_slow = x_slow + K * (a[t] - x_slow)
        P_s = (1 - K) * (P_s + Q_s)
        b_slow[t] = x_slow

    # 4. Filter Difference
    c_diff = b_fast - b_slow

    # 5. STRICT ANTI-FLIP STATE ENGINE
    signals = []
    current_state = "⚪ SIDEWAYS"
    
    upper_barrier = barrier_gate
    lower_barrier = -barrier_gate

    for val in c_diff:
        if current_state == "⚪ SIDEWAYS":
            if val > upper_barrier:
                current_state = "🟢 BUY TREND"
            elif val < lower_barrier:
                current_state = "🔴 SELL TREND"
        
        elif current_state == "🟢 BUY TREND":
            if val < lower_barrier:
                current_state = "🔴 SELL TREND"
                
        elif current_state == "🔴 SELL TREND":
            if val > upper_barrier:
                current_state = "🟢 BUY TREND"
                
        signals.append(current_state)

    # 6. Table Layout
    df_table = pd.DataFrame({
        'a: Nifty Close': np.round(a, 2),
        'b_fast (Hyper)': np.round(b_fast, 2),
        'b_slow (Lazy)': np.round(b_slow, 2),
        'c_diff (Spread)': np.round(c_diff, 2),
        'Final Locked Signal': signals
    }, index=timestamps)

    # 7. Reverse Table for Real-time analysis
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("Tuned Multipliers Applied Successfully!")
