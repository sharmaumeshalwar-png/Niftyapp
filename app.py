import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🚀 Nifty & India VIX 5-Minute Reliable Dashboard")

st.write("Fixed Architecture: Syntax Line 175 Fixed Perfectly | Trailing Window Sync Corrected | Q=0.50...")

# 1. OPTIMIZED FUNCTION WITH HOLIDAY RETRY ENGINE
@st.cache_data(ttl=1800)
def load_five_minute_data():
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=59)
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    nifty_raw = yf.download('^NSEI', start=start_str, end=end_str, interval='5m')
    vix_raw = yf.download('^INDIAVIX', start=start_str, end=end_str, interval='5m')
    
    # FALLBACK ENGINE: Weekend/Holiday backup
    if nifty_raw.empty or vix_raw.empty:
        start_dt_fallback = end_dt - timedelta(days=60)
        start_str_f = start_dt_fallback.strftime('%Y-%m-%d')
        nifty_raw = yf.download('^NSEI', start=start_str_f, end=end_str, interval='5m')
        vix_raw = yf.download('^INDIAVIX', start=start_str_f, end=end_str, interval='5m')

    if nifty_raw.empty or vix_raw.empty:
        return None

    # Flattening MultiIndex Columns safely
    nifty_raw.columns = [f"{col[0]}_nifty" if isinstance(col, tuple) else f"{col}_nifty" for col in nifty_raw.columns]
    vix_raw.columns = [f"{col[0]}_vix" if isinstance(col, tuple) else f"{col}_vix" for col in vix_raw.columns]

    # Structuring matrix
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw['High_nifty']
    nifty_df['Low_nifty'] = nifty_raw['Low_nifty']
    nifty_df['Open_nifty'] = nifty_raw['Open_nifty']
    nifty_df['Close_nifty'] = nifty_raw['Close_nifty']

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw['High_vix']
    vix_df['Low_vix'] = vix_raw['Low_vix']
    vix_df['Close_vix'] = vix_raw['Close_vix']

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H:%M')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H:%M')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute clean data engine
combined_data = load_five_minute_data()

if combined_data is None or len(combined_data) == 0:
    st.warning("⚠️ Market Data currently syncing or API connection taking time. Click 'Rerun' from the top right menu to refresh the stream.")
else:
    # Array conversions
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

    # 2. CLASSIC GAP DE-TRENDING ENGINE FOR 5-MINUTE CANDLES
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 3.0:  
                cumulative_gap += gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

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

    nifty_high_real = b_nifty_high + cumulative_gap
    nifty_low_real = b_nifty_low + cumulative_gap

    # 5. INDIA VIX KALMAN FILTERS
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
