import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide") # Dashboard space ko expand karne ke liye
st.title("Nifty & India VIX Frozen Data Intelligence Engine")

st.write("Status: Data Matrix Frozen | Cache Lock Active (1 Hour TTL) | Q=0.50 | Dual Independent Hints...")

# 1. FUNCTION TO DOWNLOAD AND FREEZE/CACHE DATA
@st.cache_data(ttl=3600)  # Data ko 1 ghante tak memory mein freeze rakhega
def load_frozen_data():
    # Data download from July 2024 to June 2026 frame
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2026-07-01', interval='1h')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2026-07-01', interval='1h')
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # yfinance MultiIndex flattening
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in nifty_raw.columns]
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vix_raw.columns]
        
    # Extracting core series
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'High' in c][0]]
    nifty_df['Low_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Low' in c][0]]
    nifty_df['Open_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Open' in c][0]]
    nifty_df['Close_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Close' in c][0]]

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw[[c for c in vix_raw.columns if 'High' in c][0]]
    vix_df['Low_vix'] = vix_raw[[c for c in vix_raw.columns if 'Low' in c][0]]
    vix_df['Close_vix'] = vix_raw[[c for c in vix_raw.columns if 'Close' in c][0]]

    # Converting index to local timezone naive string for flawless matching
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Core Synchronized Merge
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined

# Execute the frozen download engine
combined_data = load_frozen_data()

if combined_data is None:
    st.error("Data source sync failed. Check Network connectivity.")
else:
    # Safe Array Flattening from the static dataframe
    n_high = combined_data['High_nifty'].values.flatten()
    n_low = combined_data['Low_nifty'].values.flatten()
    n_open = combined_data['Open_nifty'].values.flatten()
    n_close = combined_data['Close_nifty'].values.flatten()
    
    v_high = combined_data['High_vix'].values.flatten()
    v_low = combined_data['Low_vix'].values.flatten()
    v_close = combined_data['Close_vix'].values.flatten()
    
    num_steps = len(combined_data)
    timestamps = combined_data.index.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. NIFTY GAP DE-TRENDING ENGINE
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 3. NIFTY KALMAN FILTERS
    b_nifty_high = np.zeros(num_steps)
    x_n_high = n_high_adj[0]
    P_nh, Q_nh, R_nh = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nh + Q_nh) / (P_nh + Q_nh + R_nh)
        x_n_high = x_n_high + K * (n_high_adj[t] - x_n_high)
        P_nh = (1 - K) * (P_nh + Q_nh)
        b_nifty_high[t] = x_n_high

    b_nifty_low = np.zeros(num_steps)
    x_n_low = n_low_adj[0]
    P_nl, Q_nl, R_nl = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nl + Q_nl) / (P_nl + Q_nl + R_nl)
        x_n_low = x_n_low + K * (n_low_adj[t] - x_n_low)
        P_nl = (1 - K) * (P_nl + Q_nl)
        b_nifty_low[t] = x_n_low

    nifty_high_real = b_nifty_high + cumulative_gap
    nifty_low_real = b_nifty_low + cumulative_gap

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

    # 5. INDEPENDENT NIFTY SIGNALS (Frozen logic maps)
    nifty_signals = []
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            nifty_signals.append("🟢 BUY (Nifty Cross)")
        elif n_close[t] < nifty_low_real[t]:
            nifty_signals.append("🔴 SELL (Nifty Break)")
        else:
            nifty_signals.append("⚪ SIDEWAYS")

    # 6. INDEPENDENT INDIA VIX SIGNALS
    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 SELL NIFTY (VIX High Cross)")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 BUY NIFTY (VIX Low Cross)")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # 7. MATRICES COMPILE WITH ABSOLUTE COMPACT VALUE STRING LOCKS
    df_table = pd.DataFrame({
        'Nifty Close': nifty_close_formatted = [f"{x:.2f}" for x in n_close],
        'Nifty High K': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K': [f"{x:.2f}" for x in nifty_low_real],
        '📈 NIFTY SIGNAL': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX High K': [f"{x:.2f}" for x in vifty_high],
        'VIX Low K': [f"{x:.2f}" for x in vifty_low],
        '🔥 VOLATILITY SIGNAL': vix_signals
    }, index=timestamps)

    # 8. RENDER IMMUTABLE VIEW (Latest on top)
    st.dataframe(df_table.iloc[::-1], use_container_width=True)
    st.button("🔄 Clear Cache & Force Refresh Data") # Manual update feature
