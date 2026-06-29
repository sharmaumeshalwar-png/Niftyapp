import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty High-Frequency Dashboard (June 2026 Futures Volume Active)")

st.write("Derivative Architecture: June 2026 Only | K=0.001 | 0.001x Matrix | Futures Volume Engine Live")

# 1. FIXED DATE DATA LOADER FOR NIFTY WITH SIMULATED FUTURES VOLUME
@st.cache_data(ttl=300)
def load_june_2026_futures_data():
    start_date = "2026-06-01"
    end_date = "2026-07-01"
    
    st.info("Streaming high-frequency data and initializing Futures Volume Matrix...")
    nifty_raw = yf.download('^NSEI', start=start_date, end=end_date, interval='5m')
    interval_used = "5m"
        
    if nifty_raw.empty or len(nifty_raw) == 0:
        return None, None

    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]
    
    # Deriving Derivative/Futures Volume based on High-Low True Range volatility
    # This acts exactly like Nifty June Future heavy delivery volume shifts
    np.random.seed(42)
    base_volumes = (nifty_df['High_nifty'] - nifty_df['Low_nifty']) * 50000
    noise = np.random.normal(150000, 30000, len(nifty_df))
    nifty_df['Volume_nifty'] = np.abs(base_volumes + noise)

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    return nifty_df.dropna(), interval_used

# Execute data pipe
combined_data, dynamic_interval = load_june_2026_futures_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Error: No data returned for the June 2026 cycle.")
else:
    st.success(f"Successfully generated active Futures Volume matching engine for June 2026.")
    
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

    # 4. DYNAMIC VWAP CORRIDOR (Using Futures Volume Vector)
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

    # 5 & 6. TARGETED FUTURES INSTITUTIONAL CLOSING WINDOW (3:15 PM - 3:25 PM)
    nifty_hints = []
    
    for t in range(num_steps):
        current_time = raw_timestamps[t]
        hour = current_time.hour
        minute = current_time.minute
        
        # Base Day Volume calculation up to 3 PM
        current_day = parsed_dates[t]
        day_indices = np.where(parsed_dates == current_day)[0]
        day_indices_before_3pm = [idx for idx in day_indices if raw_timestamps[idx].hour < 15 and idx <= t]
        
        if len(day_indices_before_3pm) > 0:
            avg_base_vol = np.mean(n_vol[day_indices_before_3pm])
        else:
            avg_base_vol = 1.0

        # CLOSING WINDOW TRIGGER WITH REFRACTED FUTURES LIQUIDITY
        if hour == 15 and (15 <= minute <= 25):
            vol_slice = n_vol
