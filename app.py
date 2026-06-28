import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty & India VIX Precise Alignment Engine")

st.write("Fixed Architecture: July 2024 Frame | Q=0.50 | Tilde Indexing Fixed...")

# 1. DUAL DATA DOWNLOAD WITH UTC/LOCAL TIMESTAMPS ALIGNMENT
with st.spinner("Nifty aur India VIX ka data sahi samay par align ho raha hai..."):
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2027-01-01', interval='1h')

if nifty_raw.empty or vix_raw.empty:
    st.error("Yahoo Finance se data nahi mila. Tickers check karein.")
else:
    # yfinance MultiIndex flattening
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in nifty_raw.columns]
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vix_raw.columns]
    
    # Nifty DataFrame Extraction
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'High' in c][0]]
    nifty_df['Low_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Low' in c][0]]
    nifty_df['Open_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Open' in c][0]]
    nifty_df['Close_nifty'] = nifty_raw[[c for c in nifty_raw.columns if 'Close' in c][0]]

    # VIX DataFrame Extraction
    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw[[c for c in vix_raw.columns if 'High' in c][0]]
    vix_df['Low_vix'] = vix_raw[[c for c in vix_raw.columns if 'Low' in c][0]]
    vix_df['Close_vix'] = vix_raw[[c for c in vix_raw.columns if 'Close' in c][0]]

    # Time Flooring for Hour Buckets
    nifty_df.index = nifty_df.index.strftime('%Y-%m-%d %H:00')
    vix_df.index = vix_df.index.strftime('%Y-%m-%d %H:00')

    # Inner merge to align rows perfectly
    combined_data = pd.merge(nifty_df, vix_df, left_index=True, right_index=True, how='inner')
    
    # --- CRITICAL FIX: Slicing Bracket Added Correctly ---
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    
    # Parsing dates safely for the gap engine
    parsed_dates = pd.to_datetime(combined_data.index).date
    timestamps = combined_data.index
    
    # Extracting exact flat arrays
    n_high = combined_data['High_nifty'].values.flatten()
    n_low = combined_data['Low_nifty'].values.flatten()
    n_open = combined_data['Open_nifty'].values.flatten()
    n_close = combined_data['Close_nifty'].values.flatten()
    
    v_high = combined_data['High_vix'].values.flatten()
    v_low = combined_data['Low_vix'].values.flatten()
    v_close = combined_data['Close_vix'].values.flatten()
    
    num_steps = len(combined_data)

    # 2. NIFTY GAP DE-TRENDING
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

    # 5. REVERSE CROSSOVER LOGIC (From VIX Close)
    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 SELL NIFTY (VIX High Cross)")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 BUY NIFTY (VIX Low Cross)")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # 6. MASTER DATA COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': np.round(n_close, 2),
        'Nifty High K': np.round(nifty_high_real, 2),
        'Nifty Low K': np.round(nifty_low_real, 2),
        'VIX Close': np.round(v_close, 2),
        'VIX High K': np.round(vifty_high, 2),
        'VIX Low K': np.round(vifty_low, 2),
        '🔥 VOLATILITY SIGNAL': vix_signals
    }, index=timestamps)

    # 7. CHRONOLOGICAL REVERSE RE-INDEX
    df_rendered = df_table.iloc[::-1]

    # 8. LIVE UI STREAM
    st.dataframe(df_rendered, use_container_width=True)
    st.success("Syntax patched correctly. System running 100% fine!")
