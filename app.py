import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("⚡ Nifty 5-Minute High-Frequency Engine")
st.write("Dynamic 5-Min Multi-Vector Analytics Engine | Volatility Bands & Momentum Filters Included")

# 1. DYNAMIC ROLLING 5-MIN DATA LOADER (Prevents yfinance 60-day limit crash)
@st.cache_data(ttl=300)
def load_5min_engine_data():
    # Today is June 30, 2026. Rolling back ~50 days safely within the 60-day limit for 5m interval.
    end_date = datetime(2026, 7, 1)
    start_date = end_date - timedelta(days=50)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    st.info(f"Streaming high-density 5-minute vectors from {start_str} to {end_str}...")
    nifty_raw = yf.download('^NSEI', start=start_str, end=end_str, interval='5m')
    
    if nifty_raw.empty or len(nifty_raw) == 0:
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

if combined_data is None or len(combined_data) == 0:
    st.error("Severe Error: 5-Minute Data pipeline returned empty array. Check network or date constraints.")
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

    # 2. CONTINUOUS INTRADAY GAP CALCULATOR (Adapted for M5 boundaries)
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    historical_gaps = np.zeros(num_steps)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 3.0: # Tightened threshold for 5m chart openings
                cumulative_gap += gap
        historical_gaps[t] = cumulative_gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 3. HIGH-FREQUENCY KALMAN FILTRATION ENGINE (K tuned to 0.005 for 5-min responsiveness)
    b_high = np.zeros(num_steps)
    b_low = np.zeros(num_steps)
    b_high[0], b_low[0] = n_high_adj[0], n_low_adj[0]
    K_factor = 0.005 # Faster tracking for micro-trends

    for t in range(1, num_steps):
        b_high[t] = b_high[t-1] + K_factor * (n_high_adj[t] - b_high[t-1])
        b_low[t] = b_low[t-1] + K_factor * (n_low_adj[t] - b_low[t-1])

    fixed_mid = (b_high + b_low) / 2.0
    mid_real_line = fixed_mid + historical_gaps
    
    # 5-Min True Range (ATR 20) Volatility Envelope
    atr = np.zeros(num_steps)
    atr[0] = n_high[0] - n_low[0]
    for t in range(1, num_steps):
        tr = max(n_high[t] - n_low[t], abs(n_high[t] - n_close[t-1]), abs(n_low[t] - n_close[t-1]))
        atr[t] = (atr[t-1] * 19 + tr) / 20  
    
    # Scalping corridor bands
    kalman_upper = mid_real_line + (0.75 * atr)
    kalman_lower = mid_real_line - (0.75 * atr)

    # 4. INTRAD
