import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty Gap-Normalized No-Flip Kalman Filter")

# Sidebar Controls
st.sidebar.header("Advanced Gap & Multiplier Controls")
fast_multiplier = st.sidebar.slider("Fast Kalman Sensitivity (Q)", 0.1, 2.0, 1.2, step=0.1)
slow_multiplier = st.sidebar.slider("Slow Kalman Smoothness (R)", 5.0, 50.0, 25.0, step=1.0)
gap_threshold = st.sidebar.slider("Gap Protection Threshold (Points)", 10.0, 100.0, 30.0, step=5.0)
barrier_gate = st.sidebar.slider("Anti-Flip Barrier Gate", 5.0, 50.0, 20.0, step=1.0)

# 1. Data Download
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

if data.empty:
    st.error("Data nahi mil pa raha hai.")
else:
    # Pandas index se actual Date aur Time alag nikalna taaki gap detect ho sake
    data['Date'] = data.index.date
    data['Time'] = data.index.time
    
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    a = data['Close'].values.flatten()
    o = data['Open'].values.flatten()
    num_steps = len(a)

    # 2. GAP DETECTION ENGINE
    # Har row ke liye check karenge ki kya yeh din ki pehli candle hai aur gap kitna hai
    b_fast = np.zeros(num_steps)
    b_slow = np.zeros(num_steps)
    
    x_fast = a[0]
    x_slow = a[0]
    P_f, P_s = 1.0, 1.0

    for t in range(num_steps):
        # Default Parameters
        Q_f = fast_multiplier
        R_f = 1.0
        Q_s = 0.01
        R_s = slow_multiplier
        
        # Gap Adjuster Logic: Agar current open pichle close se bhot door hai
        if t > 0:
            current_date = data['Date'].iloc[t]
            prev_date = data['Date'].iloc[t-1]
            
            # Agar din badal raha hai (Opening Candle)
            if current_date != prev_date:
                gap = abs(o[t] - a[t-1])
                
                # Agar gap predefined threshold se bada hai, toh filter ki sensitivity crash karo
                if gap > gap_threshold:
                    R_f = R_f * 5  # Fast filter ko thoda shant karo
                    R_s = R_s * 10 # Slow filter ko ekdum freeze kar do taaki gap absorbs ho sake

        # 3. FAST KALMAN EXECUTION
        K_f = (P_f + Q_f) / (P_f + Q_f + R_f)
        x_fast = x_fast + K_f * (a[t] - x_fast)
        P_f = (1 - K_f) * (P_f + Q_f)
        b_fast[t] = x_fast

        # 4. SLOW KALMAN EXECUTION
        K_s = (P_s + Q_s) / (P_s + Q_s + R_s)
        x_slow = x_slow + K_s * (a[t] - x_slow)
        P_s = (1 - K_s) * (P_s + Q_s)
        b_slow[t] = x_slow

    # 5. Difference Spread
    c_diff = b_fast - b_slow

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

    # 7. Table Setup
    df_table = pd.DataFrame({
        'a: Nifty Close': np.round(a, 2),
        'b_fast (Gap-Protected)': np.round(b_fast, 2),
        'b_slow (Gap-Protected)': np.round(b_slow, 2),
        'c_diff (Spread)': np.round(c_diff, 2),
        'Final Stable Signal': signals
    }, index=timestamps)

    # Show latest on top
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("Mathematical Gap-Normalization Activated!")
