import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty & India VIX Color-Coded Kalman Dashboard")

st.write("Fixed Architecture: Style Index Mismatch Patched | Q=0.50 | 100% Stable...")

# 1. FUNCTION TO DOWNLOAD, ALIGN AND CLEAN DATA
@st.cache_data(ttl=3600)
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2027-01-01', interval='1h')
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # yfinance MultiIndex flattening
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in nifty_raw.columns]
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vix_raw.columns]
        
    # Extracting core series safely
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'High' in c][0]]
    nifty_df['Low_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Low' in c][0]]
    nifty_df['Open_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Open' in c][0]]
    nifty_df['Close_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Close' in c][0]]

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw[[c for c in vix_raw.columns if 'High' in c][0]]
    vix_df['Low_vix'] = vix_raw[[c for c in vix_raw.columns if 'Low' in c][0]]
    vix_df['Close_vix'] = vix_raw[[c for c in vix_raw.columns if 'Close' in c][0]]

    # Timezone removal
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
    
    # Drop missing values
    combined = combined.dropna()
    return combined

# Execute data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Data matrix mapping fail ho gayi.")
else:
    # Arrays extraction
    n_high = combined_data['High_nifty'].values.astype(float).flatten()
    n_low = combined_data['Low_nifty'].values.astype(float).flatten()
    n_open = combined_data['Open_nifty'].values.astype(float).flatten()
    n_close = combined_data['Close_nifty'].values.astype(float).flatten()
    
    v_high = combined_data['High_vix'].values.astype(float).flatten()
    v_low = combined_data['Low_vix'].values.astype(float).flatten()
    v_close = combined_data['Close_vix'].values.astype(float).flatten()
    
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

    # 5. INDEPENDENT NIFTY SIGNALS + REAL DIRECTION TRACKING
    nifty_signals = []
    nifty_colors = []
    current_signal = "⏳ INITIALIZING"
    
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            current_signal = "🟢 BUY (Nifty Cross)"
            nifty_colors.append("background-color: #2e7d32; color: white; font-weight: bold;")
        elif n_close[t] < nifty_low_real[t]:
            current_signal = "🔴 SELL (Nifty Break)"
            nifty_colors.append("background-color: #c62828; color: white; font-weight: bold;")
        else:
            nifty_colors.append("background-color: #ef6c00; color: white; font-weight: bold;")
            
        nifty_signals.append(current_signal)

    # 6. INDEPENDENT INDIA VIX SIGNALS + DIRECTION TRACKING
    vix_signals = []
    vix_colors = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 SELL NIFTY (VIX High Cross)")
            vix_colors.append("background-color: #c62828; color: white; font-weight: bold;")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 BUY NIFTY (VIX Low Cross)")
            vix_colors.append("background-color: #2e7d32; color: white; font-weight: bold;")
        else:
            vix_signals.append("⚪ SIDEWAYS")
            vix_colors.append("background-color: #ef6c00; color: white; font-weight: bold;")

    # 7. EXPLICIT CHRONOLOGICAL BASE TABLE
    df_base = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty High K': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K': [f"{x:.2f}" for x in nifty_low_real],
        '📈 NIFTY HINT': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX High K': [f"{x:.2f}" for x in vifty_high],
        'VIX Low K': [f"{x:.2f}" for x in vifty_low],
        '🔥 VOLATILITY HINT': vix_signals,
        '_nifty_style': nifty_colors, # Hidden tracking elements
        '_vix_style': vix_colors
    }, index=timestamps)

    # Reverse slicing before styling injection to prevent line 165 shift
    df_reversed = df_base.iloc[::-1]

    #
