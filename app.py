import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🛡️ Nifty 5-Minute High-Frequency Engine")
st.write("Dynamic 5-Min Multi-Vector Analytics Engine | Production Safe v2.2")

# 1. DYNAMIC ROLLING 5-MIN DATA LOADER WITH EMPTY FALLBACK
@st.cache_data(ttl=60)  
def load_5min_engine_data():
    end_date = datetime(2026, 7, 1)
    start_date = end_date - timedelta(days=50)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    st.info(f"Streaming high-density 5-minute vectors from {start_str} to {end_str}...")
    nifty_raw = yf.download('^NSEI', start=start_str, end=end_str, interval='5m')
    
    if nifty_raw.empty or len(nifty_raw) < 25:
        start_date_fb = end_date - timedelta(days=55)
        nifty_raw = yf.download('^NSEI', start=start_date_fb.strftime('%Y-%m-%d'), end=end_str, interval='5m')
        
    if nifty_raw.empty:
        return None

    # Cross-Section Structural Extraction
    df = pd.DataFrame(index=nifty_raw.index)
    df['High'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0] if 'High' in nifty_raw.columns.levels[0] else nifty_raw['High']
    df['Low'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0] if 'Low' in nifty_raw.columns.levels[0] else nifty_raw['Low']
    df['Open'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0] if 'Open' in nifty_raw.columns.levels[0] else nifty_raw['Open']
    df['Close'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0] if 'Close' in nifty_raw.columns.levels[0] else nifty_raw['Close']
    
    # 5-Minute Intraday Noise Volume Modeling
    np.random.seed(42)
    vol_base = (df['High'] - df['Low']) * 25000
    noise = np.random.normal(50000, 10000, len(df))
    df['Volume'] = np.abs(vol_base + noise)
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()

# Execute high-frequency pipeline
combined_data = load_5min_engine_data()

if combined_data is None or len(combined_data) < 20:
    st.error("🚨 Severe Error: API Server returned insufficient bars. Standby for market data stream to initialize.")
else:
    st.success(f"Successfully loaded {len(combined_data)} high-frequency 5-min intervals.")
    
    # Pure Linear Arrays
    n_high = combined_data['High'].to_numpy(dtype=float)
    n_low = combined_data['Low'].to_numpy(dtype=float)
    n_open = combined_data['Open'].to_numpy(dtype=float)
    n_close = combined_data['Close'].to_numpy(dtype=float)
    n_vol = combined_data['Volume'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    parsed_dates = combined_data.index.date
    timestamps = combined_data.index

    # 2. CONTINUOUS INTRADAY GAP CALCULATOR 
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    historical_gaps = np.zeros(num_steps, dtype=float)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 3.0: 
                cumulative_gap += gap
        historical_gaps[t] = cumulative_gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 3. HIGH-FREQUENCY KALMAN FILTRATION ENGINE 
    b_high = np.zeros(num_steps, dtype=float)
    b_low = np.zeros(num_steps, dtype=float)
    b_high[0], b_low[0] = n_high_adj[0], n_low_adj[0]
    K_factor = 0.005 

    for t in range(1, num_steps):
        b_high[t] = b_high[t-1] + K_factor * (n_high_adj[t] - b_high[t-1])
        b_low[t] = b_low[t-1] + K_factor * (n_low_adj[t] - b_low[t-1])

    fixed_mid = (b_high + b_low) / 2.0
    mid_real_line = fixed_mid + historical_gaps
    
    # 5-Min True Range (ATR 20) Volatility Envelope
    atr = np.zeros(num_steps, dtype=float)
    atr[0] = n_high[0] - n_low[0]
    for t in range(1, num_steps):
        tr = max(n_high[t] - n_low[t], abs(n_high[t] - n_close[t-1]), abs(n_low[t] - n_close[t-1]))
        atr[t] = (atr[t-1] * 19 + tr) / 20  
    
    kalman_upper = mid_real_line + (0.75 * atr)
    kalman_lower = mid_real_line - (0.75 * atr)

    # 4. INTRADAY SESSION CUMULATIVE VWAP 
    vwap = np.zeros(num_steps, dtype=float)
    cum_pv = 0.0
    cum_vol = 0.0
    
    for t in range(num_steps):
        if t == 0 or parsed_dates[t] != parsed_dates[t-1]:
            cum_pv = ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol = n_vol[t] if n_vol[t] > 0 else 1.0
        else:
            cum_pv += ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol += n_vol[t]
        vwap[t] = cum_pv / cum_vol

    # 5. FAST MOMENTUM TRACKER - RSI 14 
    rsi = np.full(num_steps, 50.0, dtype=float) 
    if num_steps > 15:
        gains = np.zeros(num_steps, dtype=float)
        losses = np.zeros(num_steps, dtype=float)
        
        for t in range(1, num_steps):
            diff = n_close[t] - n_close[t-1]
            gains[t] = diff if diff > 0 else 0.0
            losses[t] = -diff if diff
