import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🦅 Nifty & India VIX Daily Close-Only Dashboard")

st.write("Fixed Architecture: Daily Continuous Interval | Close Price Kalman Filter Only | Q=0.50 | Pure Signals...")

# 1. OPTIMIZED FUNCTION FOR DAILY DATA FETCHING (Close Prices Only)
@st.cache_data(ttl=3600) 
def load_daily_close_data():
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=730)  # Pichle 2 saal ka historical data
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    # Fetching Daily Data
    nifty_raw = yf.download('^NSEI', start=start_str, end=end_str, interval='1d', progress=False)
    vix_raw = yf.download('^INDIAVIX', start=start_str, end=end_str, interval='1d', progress=False)
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # Flattening MultiIndex Columns safely
    nifty_raw.columns = [f"{col[0]}_nifty" if isinstance(col, tuple) else f"{col}_nifty" for col in nifty_raw.columns]
    vix_raw.columns = [f"{col[0]}_vix" if isinstance(col, tuple) else f"{col}_vix" for col in vix_raw.columns]

    # Mapping from flattened structure (High and Low removed)
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['Open_nifty'] = nifty_raw['Open_nifty']
    nifty_df['Close_nifty'] = nifty_raw['Close_nifty']

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['Close_vix'] = vix_raw['Close_vix']

    # Clean timezone alignment
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Synchronized Join
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Date_x'] if 'Date_x' in combined.columns else combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute clean data engine
combined_data = load_daily_close_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Error loading Daily historical data stream. Please check connection.")
else:
    # Array Extractions
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    v_close = combined_data['Close_vix'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    timestamps = combined_data.index.strftime('%Y-%m-%d')
    parsed_dates = combined_data.index.date

    # 2. GAP DE-TRENDING ENGINE FOR DAILY CLOSE ONLY
    n_close_adj = np.copy(n_close)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        n_close_adj[t] = n_close[t] - cumulative_gap

    # 3. NIFTY CLOSE KALMAN FILTER (Q = 0.50)
    b_nifty_close = np.zeros(num_steps)
    x_n_close = n_close_adj[0]
    P_nc, Q_nc, R_nc = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_nc + Q_nc) / (P_nc + Q_nc + R_nc)
        x_n_close = x_n_close + K * (n_close_adj[t] - x_n_close)
        P_nc = (1 - K) * (P_nc + Q_nc)
        b_nifty_close[t] = x_n_close

    nifty_close_real = b_nifty_close + cumulative_gap

    # 4. INDIA VIX CLOSE KALMAN FILTER
    vifty_close = np.zeros(num_steps)
    x_v_close = v_close[0]
    P_vc, Q_vc, R_vc = 1.0, 0.50, 1.0
    for t in range(num_steps):
        K = (P_vc + Q_vc) / (P_vc + Q_vc + R_vc)
        x_v_close = x_v_close + K * (v_close[t] - x_v_close)
        P_vc = (1 - K) * (P_vc + Q_vc)
        vifty_close[t] = x_v_close

    # 5. CLOSE-BASED TREND SIGNALS
    nifty_signals = []
    current_signal = "⏳ INITIALIZING"
    for t in range(num_steps):
        if n_close[t] > nifty_close_real[t]:
            current_signal = "🟢 BUY"
        elif n_close[t] < nifty_close_real[t]:
            current_signal = "🔴 SELL"
        nifty_signals.append(current_signal)

    # 6. INDIA VIX CLOSE SIGNALS
    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_close[t]:
            vix_signals.append("🔴 RISK HIGH")
        elif v_close[t] < vifty_close[t]:
            vix_signals.append("🟢 RISK LOW")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # 7. DATAFRAME COMPILATION (Clean & Light View)
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty Close K': [f"{x:.2f}" for x in nifty_close_real],
        '📈 NIFTY HINT': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX Close K': [f"{x:.2f}" for x in vifty_close],
        '🔥 VOLATILITY HINT': vix_signals
    }, index=timestamps)

    df_reversed = df_table.iloc[::-1]

    def style_nifty_strict(val):
        if "BUY" in str(val):
            return "background-color: #2e7d32; color: white; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #c62828; color: white; font-weight: bold;"
        return ""

    styled_final_df = df_reversed.style.map(style_nifty_strict, subset=['📈 NIFTY HINT'])

    # 8. RENDER VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Close-Only Dashboard is running perfectly. High-Low boundaries removed successfully!")
