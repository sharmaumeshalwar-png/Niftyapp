import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty 1-Hour Kalman Channel (From 1 Jan 2024)")

st.write("Fixed Architecture: Q=0.50 Par High/Low Filters (Full 2024-2026 Data Active)...")

# 1. Data Download (1-Hour Interval Starting from 1 Jan 2024)
data = yf.download('^NSEI', start='2024-01-01', end='2027-01-01', interval='1h')

if data.empty:
    st.error("Yahoo Finance se 1 Jan 2024 se data nahi mil pa raha hai.")
else:
    data['Date'] = data.index.date
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    
    # Inputs
    price_high = data['High'].values.flatten()
    price_low = data['Low'].values.flatten()
    o = data['Open'].values.flatten()
    c_close = data['Close'].values.flatten()
    num_steps = len(c_close)

    # 2. LONG-TERM GAP DE-TRENDING FOR HIGH AND LOW
    high_adjusted = np.copy(price_high)
    low_adjusted = np.copy(price_low)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        current_date = data['Date'].iloc[t]
        prev_date = data['Date'].iloc[t-1]
        
        if current_date != prev_date:
            gap = o[t] - c_close[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        
        high_adjusted[t] = price_high[t] - cumulative_gap
        low_adjusted[t] = price_low[t] - cumulative_gap

    # 3. HIGH KALMAN FILTER (a) - Fixed Q = 0.50
    b_high = np.zeros(num_steps)
    x_high = high_adjusted[0]
    P_h, Q_h, R_h = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_h + Q_h) / (P_h + Q_h + R_h)
        x_high = x_high + K * (high_adjusted[t] - x_high)
        P_h = (1 - K) * (P_h + Q_h)
        b_high[t] = x_high

    # 4. LOW KALMAN FILTER (b) - Fixed Q = 0.50
    b_low = np.zeros(num_steps)
    x_low = low_adjusted[0]
    P_l, Q_l, R_l = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_l + Q_l) / (P_l + Q_l + R_l)
        x_low = x_low + K * (low_adjusted[t] - x_low)
        P_l = (1 - K) * (P_l + Q_l)
        b_low[t] = x_low

    # 5. Bring Back to Real Scale
    a_high_real = b_high + cumulative_gap
    b_low_real = b_low + cumulative_gap
    
    # 6. CALCULATE C = A - B (Dynamic Channel Spread)
    c_diff = a_high_real - b_low_real

    # 7. HIGH-LOW BREAKOUT SIGNAL LOGIC
    signals = []
    current_state = "⚪ SIDEWAYS"
    
    for t in range(num_steps):
        if c_close[t] > a_high_real[t]:
            current_state = "🟢 BUY (High Breakout)"
        elif c_close[t] < b_low_real[t]:
            current_state = "🔴 SELL (Low Breakout)"
        signals.append(current_state)

    # 8. Create Channel DataFrame Table
    df_table = pd.DataFrame({
        'Nifty Close': np.round(c_close, 2),
        'a: High Kalman': np.round(a_high_real, 2),
        'b: Low Kalman': np.round(b_low_real, 2),
        'c: Spread (a - b)': np.round(c_diff, 2),
        'Channel Signal': signals
    }, index=timestamps)

    # Render Table (Latest rows on Top)
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.success("Historical 2024-2026 Engine Deployed Successfully!")
