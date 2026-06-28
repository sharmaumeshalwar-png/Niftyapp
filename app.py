import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty & India VIX Dual Kalman Channel")

st.write("Fixed Architecture: High/Low Kalman Channels Independent | Price Range Aligned | Q=0.50...")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT SINGLE-COLUMN SERIES
@st.cache_data(ttl=3600)
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2027-01-01', interval='1h')
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # Safe MultiIndex Extraction using Cross-Section (.xs)
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw.xs('High', axis=1, level=0).iloc[:, 0]
    vix_df['Low_vix'] = vix_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    vix_df['Close_vix'] = vix_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    # Timezone clean-up
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Synchronized Merge
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute clean data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Data tracking framework error.")
else:
    # Pure Linear Arrays
    n_high = combined_data['High_nifty'].to_numpy(dtype=float)
    n_low = combined_data['Low_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    
    v_high = combined_data['High_vix'].to_numpy(dtype=float)
    v_low = combined_data['Low_vix'].to_numpy(dtype=float)
    v_close = combined_data['Close_vix'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    timestamps = combined_data.index.strftime('%Y-%m-%d %H:%M')

    # 2. INDEPENDENT HIGH KALMAN FILTER (Only Tracks High Prices)
    nifty_high_real = np.zeros(num_steps)
    x_n_high = n_high[0]
    P_nh, Q_nh, R_nh = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nh + Q_nh) / (P_nh + Q_nh + R_nh)
        x_n_high = x_n_high + K * (n_high[t] - x_n_high)
        P_nh = (1 - K) * (P_nh + Q_nh)
        nifty_high_real[t] = x_n_high

    # 3. INDEPENDENT LOW KALMAN FILTER (Only Tracks Low Prices)
    nifty_low_real = np.zeros(num_steps)
    x_n_low = n_low[0]
    P_nl, Q_nl, R_nl = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nl + Q_nl) / (P_nl + Q_nl + R_nl)
        x_n_low = x_n_low + K * (n_low[t] - x_n_low)
        P_nl = (1 - K) * (P_nl + Q_nl)
        nifty_low_real[t] = x_n_low

    # 4. INDIA VIX KALMAN FILTERS
    vifty_high = np.zeros(num_steps)
    x_v_high = v_high[0]
    P_vh, Q_vh, R_vh = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_vh + Q_vh) / (P_vh + Q_vh + R_vh)
        x_v_high = x_v_high + K * (v_high[t] - x_v_high)
        P_vh = (1 - K) * (P_vh + Q_vh)
        vifty_high[t] = x_v_high

    vifty_low = np.zeros(num_steps)
    x_v_low = v_low[0]
    P_vl, Q_vl, R_vl = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_vl + Q_vl) / (P_vl + Q_vl + R_vl)
        x_v_low = x_v_low + K * (v_low[t] - x_v_low)
        P_vl = (1 - K) * (P_vl + Q_vl)
        vifty_low[t] = x_v_low

    # 5. STRICT CONTINUOUS NIFTY SIGNALS (Red/Green Continuous Block)
    nifty_signals = []
    current_signal = "⏳ INITIALIZING"
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            current_signal = "🟢 BUY"
        elif n_close[t] < nifty_low_real[t]:
            current_signal = "🔴 SELL"
        nifty_signals.append(current_signal)

    # 6. INDIA VIX SIGNALS (Standard Transparent Text)
    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 RISK HIGH")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 RISK LOW")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # 7. EXPLICIT MATRICES COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty High K': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K': [f"{x:.2f}" for x in nifty_low_real],
        '📈 NIFTY HINT': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX High K': [f"{x:.2f}" for x in vifty_high],
        'VIX Low K': [f"{x:.2f}" for x in vifty_low],
        '🔥 VOLATILITY HINT': vix_signals
    }, index=timestamps)

    df_reversed = df_table.iloc[::-1]

    # STRICT RED/GREEN CELL MAP FUNCTION
    def style_nifty_strict(val):
        if "BUY" in str(val):
            return "background-color: #2e7d32; color: white; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #c62828; color: white; font-weight: bold;"
        return ""

    styled_final_df = df_reversed.style.map(style_nifty_strict, subset=['📈 NIFTY HINT'])

    # 8. RENDER VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Perfect Dual Channel Active! High and Low tracks are independently operating.")
