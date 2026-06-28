import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🚀 Pure Gap-Only Kalman Discovery Engine")

st.write("Research Framework: Running Kalman Filter STRICTORLY on Accumulated Gaps Only | Institutional Trend Discovery...")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT MULTIINDEX SERIES
@st.cache_data(ttl=3600)
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    if nifty_raw.empty:
        return None

    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H')
    nifty_df = nifty_df.reset_index()
    
    return nifty_df.dropna()

# Execute data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Data matrix loading error.")
else:
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    timestamps = combined_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M').to_numpy()
    parsed_dates = combined_data['Datetime'].dt.date.to_numpy()

    # 2. PURE OVERNIGHT GAP ISOLATION
    pure_gap_series = np.zeros(num_steps)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        pure_gap_series[t] = cumulative_gap

    # 3. KALMAN FILTER ON PURE GAP SERIES (Q = 0.50)
    gap_kalman = np.zeros(num_steps)
    x_gap = pure_gap_series[0]
    P_g, Q_g, R_g = 1.0, 0.50, 1.0
    
    for t in range(num_steps):
        K = (P_g + Q_g) / (P_g + Q_g + R_g)
        x_gap = x_gap + K * (pure_gap_series[t] - x_gap)
        P_g = (1 - K) * (P_g + Q_g)
        gap_kalman[t] = x_gap

    # 4. INSTANT SIGNAL DISCOVERY LOGIC
    # Gap momentum line jab apne Kalman Smooth Average ke upar jayegi -> Bulls Dominating Gaps
    gap_signals = []
    current_signal = "⏳ INITIALIZING"
    for t in range(num_steps):
        if pure_gap_series[t] > gap_kalman[t]:
            current_signal = "🟢 INSTITUTIONAL BUY"
        elif pure_gap_series[t] < gap_kalman[t]:
            current_signal = "🔴 INSTITUTIONAL SELL"
        gap_signals.append(current_signal)

    # 5. DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Isolated Gap Cumulative': [f"{x:.2f}" for x in pure_gap_series],
        'Gap Kalman Line': [f"{x:.2f}" for x in gap_kalman],
        '🎯 GAP TREND HINT': gap_signals
    }, index=timestamps)

    df_reversed = df_table.iloc[::-1]

    # STYLER ENGINE
    def style_gap_strict(val):
        if "BUY" in str(val):
            return "background-color: #1b5e20; color: white; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #b71c1c; color: white; font-weight: bold;"
        return ""

    styled_final_df = df_reversed.style.map(style_gap_strict, subset=['🎯 GAP TREND HINT'])

    # 6. RENDER THE DISCOVERY DATA
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Discovery Engine Activated! You are now tracking pure overnight institutional flow.")
