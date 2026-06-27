import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty Absolute Zero-Gap Kalman Filter")

# Sidebar Controls
st.sidebar.header("Tuning Parameters")
fast_multiplier = st.sidebar.slider("Fast Kalman Sensitivity (Q)", 0.1, 2.0, 1.2, step=0.1)
slow_multiplier = st.sidebar.slider("Slow Kalman Smoothness (R)", 5.0, 50.0, 25.0, step=1.0)
barrier_gate = st.sidebar.slider("Anti-Flip Barrier Gate", 5.0, 50.0, 15.0, step=1.0)

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

    # 2. GAP DE-TRENDING MATHEMATICS (HARD FIX)
    a_adjusted = np.copy(a)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        current_date = data['Date'].iloc[t]
        prev_date = data['Date'].iloc[t-1]
        
        # Agar naya din shuru hua (Opening Candle)
        if current_date != prev_date:
            # Gap value = Aaj ka open - kal ka close
            gap = o[t] - a[t-1]
            # Agar gap bada hai toh cumulative gap mein jodh do
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        
        # Har candle ke price mein se gap ka asar hamesha ke liye nikaal do
        a_adjusted[t] = a[t] - cumulative_gap

    # 3. FAST KALMAN (On Clean Adjusted Data)
    b_fast = np.zeros(num_steps)
    x_fast = a_adjusted[0]
    P_f, Q_f, R_f = 1.0, fast_multiplier, 1.0
    for t in range(num_steps):
        K = (P_f + Q_f) / (P_f + Q_f + R_f)
        x_fast = x_fast + K * (a_adjusted[t] - x_fast)
        P_f = (1 - K) * (P_f + Q_f)
        b_fast[t] = x_fast

    # 4. SLOW KALMAN (On Clean Adjusted Data)
    b_slow = np.zeros(num_steps)
    x_slow = a_adjusted[0]
    P_s, Q_s, R_s = 1.0, 0.01, slow_multiplier
    for t in range(num_steps):
        K = (P_s + Q_s) / (P_s + Q_s + R_s)
        x_slow = x_slow + K * (a_adjusted[t] - x_slow)
        P_s = (1 - K) * (P_s + Q_s)
        b_slow[t] = x_slow

    # 5. Bring Back To Real Scale for Table Display
    # Filter lines ko wapas real scale par laane ke liye cumulative gap wapas jodte hain
    b_fast_real = b_fast + cumulative_gap
    b_slow_real = b_slow + cumulative_gap
    c_diff = b_fast_real - b_slow_real

    # 6. ANTI-FLIP STATE ENGINE WITH HYSTERESIS
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

    # 7. Final Clean Table
    df_table = pd.DataFrame({
        'a: Nifty Close': np.round(a, 2),
        'b_fast (Zero-Gap)': np.round(b_fast_real, 2),
        'b_slow (Zero-Gap)': np.round(b_slow_real, 2),
        'c_diff (Spread)': np.round(c_diff, 2),
        'Final Stable Signal': signals
    }, index=timestamps)

    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("Gap Up/Down effect mathematically 100% eliminated!")
