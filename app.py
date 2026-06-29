import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty & India VIX Classic High-Low Gap Dashboard (Reverse Loop)")

st.write("Fixed Architecture: True Reverse Kalman Processing Active | Pure Gap Engine | Q=0.50 | Start: July 1, 2024...")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT MULTIINDEX SERIES (Daily Interval)
@st.cache_data(ttl=3600)
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1d')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2027-01-01', interval='1d')
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # Safe MultiIndex Extraction using Cross-Section (.xs)
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw.xs('High', axis=1, level=0).iloc[:, 0]
    vix_df['Low_vix'] = vix_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    vix_df['Close_vix'] = vix_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    # Timezone clean-up
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Synchronized Merge
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Date_x'] if 'Date_x' in combined.columns else combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Data extraction error or no daily data available for this range.")
else:
    # Pure Linear Arrays
    n_high = combined_data['High_nifty'].to_numpy(dtype=float)
    n_low = combined_data['Low_nifty'].to_numpy(dtype=float)
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    
    v_high = combined_data['High_vix'].to_numpy(dtype=float)
    v_low = combined_data['Low_vix'].to_numpy(dtype=float)
    v_close = combined_data['Close_vix'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    timestamps = combined_data.index.strftime('%Y-%m-%d')

    # 2. TRUE REVERSE LOOP GAP ENGINE (Calculating from Present to Past)
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    cumulative_gap_arr = np.zeros(num_steps)
    single_gap_tracker = np.zeros(num_steps)
    running_gap = 0.0

    # Loop runs backward from the last element down to 0
    for t in range(num_steps - 2, -1, -1):
        gap = n_open[t+1] - n_close[t]
        single_gap_tracker[t] = gap
        
        if abs(gap) > 5.0:  
            running_gap += gap
            
        cumulative_gap_arr[t] = running_gap
        n_high_adj[t] = n_high[t] - running_gap
        n_low_adj[t] = n_low[t] - running_gap

    # 3. REVERSE NIFTY HIGH KALMAN FILTER
    b_nifty_high = np.zeros(num_steps)
    x_n_high = n_high_adj[-1]  # Shuruat peeche se hogi
    P_nh, Q_nh, R_nh = 1.0, 0.50, 1.0
    for t in range(num_steps - 1, -1, -1):
        K = (P_nh + Q_nh) / (P_nh + Q_nh + R_nh)
        x_n_high = x_n_high + K * (n_high_adj[t] - x_n_high)
        P_nh = (1 - K) * (P_nh + Q_nh)
        b_nifty_high[t] = x_n_high

    # 4. REVERSE NIFTY LOW KALMAN FILTER
    b_nifty_low = np.zeros(num_steps)
    x_n_low = n_low_adj[-1]
    P_nl, Q_nl, R_nl = 1.0, 0.50, 1.0
    for t in range(num_steps - 1, -1, -1):
        K = (P_nl + Q_nl) / (P_nl + Q_nl + R_nl)
        x_n_low = x_n_low + K * (n_low_adj[t] - x_n_low)
        P_nl = (1 - K) * (P_nl + Q_nl)
        b_nifty_low[t] = x_n_low

    # Re-applying matching row gaps
    nifty_high_real = b_nifty_high + cumulative_gap_arr
    nifty_low_real = b_nifty_low + cumulative_gap_arr

    # 5. REVERSE INDIA VIX KALMAN FILTERS
    vifty_high = np.zeros(num_steps)
    x_v_high = v_high[-1]
    P_vh, Q_vh, R_vh = 1.0, 0.50, 1.0
    for t in range(num_steps - 1, -1, -1):
        K = (P_vh + Q_vh) / (P_vh + Q_vh + R_vh)
        x_v_high = x_v_high + K * (v_high[t] - x_v_high)
        P_vh = (1 - K) * (P_vh + Q_vh)
        vifty_high[t] = x_v_high

    vifty_low = np.zeros(num_steps)
    x_v_low = v_low[-1]
    P_vl, Q_vl, R_vl = 1.0, 0.50, 1.0
    for t in range(num_steps - 1, -1, -1):
        K = (P_vl + Q_vl) / (P_vl + Q_vl + R_vl)
        x_v_low = x_v_low + K * (v_low[t] - x_v_low)
        P_vl = (1 - K) * (P_vl + Q_vl)
        vifty_low[t] = x_v_low

    # 6. SIGNALS GENERATION
    nifty_signals = ["⏳ INITIALIZING"] * num_steps
    current_signal = "⏳ INITIALIZING"
    # Signals logically follow chronological flow even if processing is back-mapped
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            current_signal = "🟢 BUY"
        elif n_close[t] < nifty_low_real[t]:
            current_signal = "🔴 SELL"
        nifty_signals[t] = current_signal

    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 RISK HIGH")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 RISK LOW")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        'Nifty Open': [f"{x:.2f}" for x in n_open],
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        '⚡ Overnight Gap': [f"{x:.2f}" for x in single_gap_tracker],
        'Nifty High K': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K': [f"{x:.2f}" for x in nifty_low_real],
        '📈 NIFTY HINT': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX High K': [f"{x:.2f}" for x in vifty_high],
        'VIX Low K': [f"{x:.2f}" for x in vifty_low],
        '🔥 VOLATILITY HINT': vix_signals
    }, index=timestamps)

    # Ab hum data ko seedha render karenge kyuki processing khud reverse matrix track par chali hai
    df_reversed = df_table.iloc[::-1]

    # 8. RENDER SECURE VIEW
    st.dataframe(df_reversed, use_container_width=True)
    st.success("Ab check karo bhai! Pure code ka loop ab flipped structure me run ho raha hai.")
