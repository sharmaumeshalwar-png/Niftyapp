import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Nifty High-Frequency Institutional Flow Dashboard")

st.write("Stable HFT Engine: K=0.001 | 0.001x Matrix | Auto-Fallback Active | 3:20 PM Signal Lock")

# 1. DYNAMIC DATA LOADER WITH AUTOMATIC FALLBACK TO PREVENT BLANK SCREEN
@st.cache_data(ttl=300)
def load_safe_high_frequency_data():
    today = datetime.now()
    # Looking back last 6 days for deep data availability
    start_date = (today - timedelta(days=6)).strftime('%Y-%m-%d')
    
    st.info("Attempting to stream 1-Minute Institutional Ticks...")
    nifty_raw = yf.download('^NSEI', start=start_date, interval='1m')
    
    # FALLBACK: If 1-Min data comes empty or breaks, shift to 5-Min data instantly
    if nifty_raw.empty or len(nifty_raw) == 0:
        st.warning("1-Min stream unavailable or rate-limited. Shifting to 5-Minute High-Frequency Backup Stream...")
        nifty_raw = yf.download('^NSEI', start=start_date, interval='5m')
        interval_used = "5m"
    else:
        interval_used = "1m"
        
    if nifty_raw.empty:
        return None, None

    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]
    nifty_df['Volume_nifty'] = nifty_raw.xs('Volume', axis=1, level=0).iloc[:, 0]

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    return nifty_df.dropna(), interval_used

# Execute data pipe
combined_data, dynamic_interval = load_safe_high_frequency_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Severe Error: Live stock exchange servers returned empty array data. Please reload after a minute.")
else:
    st.success(f"Successfully loaded continuous data matrix using '{dynamic_interval}' data blocks.")
    
    # Pure Linear Arrays
    n_high = combined_data['High_nifty'].to_numpy(dtype=float)
    n_low = combined_data['Low_nifty'].to_numpy(dtype=float)
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    n_vol = combined_data['Volume_nifty'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    raw_timestamps = combined_data.index
    timestamps_formatted = raw_timestamps.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. CONTINUOUS INTRADAY GAP CALCULATOR
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    historical_gaps = np.zeros(num_steps)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:
                cumulative_gap += gap
        historical_gaps[t] = cumulative_gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 3. FILTRATION ARCHITECTURE (K = 0.001)
    b_nifty_high = np.zeros(num_steps)
    b_nifty_low = np.zeros(num_steps)
    b_nifty_high[0], b_nifty_low[0] = n_high_adj[0], n_low_adj[0]
    K_factor = 0.001

    for t in range(1, num_steps):
        b_nifty_high[t] = b_nifty_high[t-1] + K_factor * (n_high_adj[t] - b_nifty_high[t-1])
        b_nifty_low[t] = b_nifty_low[t-1] + K_factor * (n_low_adj[t] - b_nifty_low[t-1])

    fixed_mid = (b_nifty_high + b_nifty_low) / 2.0
    fixed_spread = b_nifty_high - b_nifty_low
    nifty_high_real = (fixed_mid + (fixed_spread * 0.0005)) + historical_gaps
    nifty_low_real = (fixed_mid - (fixed_spread * 0.0005)) + historical_gaps
    mid_real_line = (nifty_high_real + nifty_low_real) / 2.0

    # 4. DYNAMIC VWAP CORRIDOR
    vwap = np.zeros(num_steps)
    cum_pv = 0.0
    cum_vol = 0.0
    
    for t in range(num_steps):
        if t == 0 or parsed_dates[t] != parsed_dates[t-1]:
            cum_pv = ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol = n_vol[t] if n_vol[t] > 0 else 1.0
        else:
            cum_pv += ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol += n_vol[t]
        vwap[t] = cum_pv / cum_vol if cum_vol > 0 else n_close[t]

    # 5 & 6. TARGETED INSTITUTIONAL 3:20 PM PATTERN MATCHING
    nifty_hints = []
    
    for t in range(num_steps):
        current_time = raw_timestamps[t]
        hour = current_time.hour
        minute = current_time.minute
        
        # Base Day Volume calculation up to 3 PM
        current_day = parsed_dates[t]
        day_indices = np.where(parsed_dates == current_day)[0]
        day_indices_before_3pm = [idx for idx in day_indices if raw_timestamps[idx].hour
