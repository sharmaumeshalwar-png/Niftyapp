import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime

st.title("Nifty 5-Min Kalman High-Low Channel")

# Sidebar for 5-Min Fine Tuning
st.sidebar.header("5-Min Channel Calibration")
test_q = st.sidebar.slider("Multiplier (Q) [Default 0.50]", 0.05, 1.50, 0.50, step=0.05)

st.write(f"Running 5-Minute Intraday Engine from 1 April 2026 | Current Q: {test_q}")

# 1. Download 5-Minute Data
# Note: yfinance allows max 60 days back for 5m interval, which safely covers April 2026 onwards.
data = yf.download('^NSEI', start='2026-04-01', end='2026-07-01', interval='5m')

if data.empty:
    st.error("Yahoo Finance se 5-minute ka data nahi mil pa raha hai. Date range check karein.")
else:
    data['Date'] = data.index.date
    # 5-min ke liye date aur time dono display karenge
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    
    price_high = data['High'].values.flatten()
    price_low = data['Low'].values.flatten()
    o = data['Open'].values.flatten()
    c_close = data['Close'].values.flatten()
    num_steps = len(c_close)

    # 2. 5-MIN CUMULATIVE GAP ENGINE
    high_adjusted = np.copy(price_high)
    low_adjusted = np.copy(price_low)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        current_date = data['Date'].iloc[t]
        prev_date = data['Date'].iloc[t-1]
        
        # Din badalte hi pehli 5-min candle par gap calculate hoga
        if current_date != prev_date:
            gap = o[t] - c_close[t-1]
            if abs(gap) > 3.0:  # 5-min par micro gaps ko bhi trace karenge
                cumulative_gap += gap
        
        high_adjusted[t] = price_high[t] - cumulative_gap
        low_adjusted[t] = price_low[t] - cumulative_gap

    # 3. HIGH KALMAN FILTER (a)
    b_high = np.zeros(num_steps)
    x_high = high_adjusted[0]
    P_h, Q_h, R_h = 1.0, test_q, 1.0
    for t in range(num_steps):
        K = (P_h + Q_h) / (P_h + Q_h + R_h)
        x_high = x_high + K * (high_adjusted[t] - x_high)
        P_h = (1 - K) * (P_h + Q_h)
        b_high[t] = x_high

    # 4. LOW KALMAN FILTER (b)
    b_low = np.zeros(num_steps)
    x_low = low_adjusted[0]
    P_l, Q_l, R_l = 1.0, test_q, 1.0
    for t in range(num_steps):
        K = (P_l + Q_l) / (P_l + Q_l + R_l)
        x_low = x_low + K * (low_adjusted[t] - x_low)
        P_l = (1 - K) * (P_l + Q_l)
        b_low[t] = x_low

    # 5. Reverse de-trending to real scale
    a_high_real = b_high + cumulative_gap
    b_low_real = b_low + cumulative_gap
    
    # 6. Dynamic Spread C = A - B
    c_diff = a_high_real - b_low_real

    # 7. Strict Breakout Signal Engine
    signals = []
    current_state = "⚪ SIDEWAYS"
    
    for t in range(num_steps):
        if c_close[t] > a_high_real[t]:
            current_state = "🟢 BUY (High Breakout)"
        elif c_close[t] < b_low_real[t]:
            current_state = "🔴 SELL (Low Breakout)"
        signals.append(current_state)

    # 8. Render Table
    df_table = pd.DataFrame({
        'Nifty Close': np.round(c_close, 2),
        'a: High Kalman': np.round(a_high_real, 2),
        'b: Low Kalman': np.round(b_low_real, 2),
        'c: Spread (a - b)': np.round(c_diff, 2),
        '5-Min Signal': signals
    }, index=timestamps)

    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("5-Minute Intraday Kalman Channel Active!")
