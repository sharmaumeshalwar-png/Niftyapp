import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty & India VIX Classic High-Low Gap Dashboard")

st.write("Fixed Architecture: 5-Min Kalman Channels Active | Max 60D Intraday Data Frozen | Q=0.50 | Pure Gap Engine Live...")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT MULTIINDEX SERIES (Auto-Frozen to Max 60 Days)
@st.cache_data(ttl=300)
def load_frozen_data():
    # Note: 5m interval ke liye Yahoo Finance maximum pichle 60 days ka data hi deta hai.
    # Isliye humne period='60d' set kiya hai taaki app kabhi blank na ho.
    nifty_raw = yf.download('^NSEI', period='60d', interval='5m')
    vix_raw = yf.download('^INDIAVIX', period='60d', interval='5m')
    
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
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H:%M')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H:%M')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Synchronized Merge
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Yahoo Finance server did not return intraday data. Please refresh or try again in a moment.")
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
    timestamps = combined_data.index.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. PURE GAP DE-TRENDING ENGINE
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    cumulative_gap_arr = np.zeros(num_steps)
    running_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:  
                running_gap += gap
        cumulative_gap_arr[t] = running_gap
        n_high_adj[t] = n_high[t] - running_gap
        n_low_adj[t] = n_low[t] - running_gap

    # 3. NIFTY HIGH KALMAN FILTER (Q = 0.50)
    b_nifty_high = np.zeros(num_steps)
    x_n_high = n_high_adj[0]
    P_nh, Q_nh, R_nh = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nh + Q_nh) / (P_nh + Q_nh + R_nh)
        x_n_high = x_n_high + K * (n_high_adj[t] - x_n_high)
        P_nh = (1 - K) * (P_nh + Q_nh)
        b_nifty_high[t] = x_n_high

    # 4. NIFTY LOW KALMAN FILTER (Q = 0.50)
    b_nifty_low = np.zeros(num_steps)
    x_n_low = n_low_adj[0]
    P_nl, Q_nl, R_nl = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nl + Q_nl) / (P_nl + Q_nl + R_nl)
        x_n_low = x_n_low + K * (n_low_adj[t] - x_n_low)
        P_nl = (1 - K) * (P_nl + Q_nl)
        b_nifty_low[t] = x_n_low

    # Re-applying matching row gaps properly
    nifty_high_real = b_nifty_high + cumulative_gap_arr
    nifty_low_real = b_nifty_low + cumulative_gap_arr

    # 5. INDIA VIX KALMAN FILTERS
    vifty_high = np.zeros(num_steps)
    x_v_high = v_high[0]
    P_vh, Q_vh, R_vh = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_vh + Q
