import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Nifty 1-Min Institutional Flow Dashboard")

st.write("Institutional Grid: K=0.001 | 0.001x Matrix | 3:20 PM VWAP & Volume Surge Lock")

# 1. FETCH 1-MINUTE HIGH FREQUENCY DATA
@st.cache_data(ttl=300)
def load_1min_institutional_data():
    # 1-min data availability is limited to last few days
    today = datetime.now()
    start_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')
    
    nifty_raw = yf.download('^NSEI', start=start_date, interval='1m')
    
    if nifty_raw.empty:
        return None

    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]
    nifty_df['Volume_nifty'] = nifty_raw.xs('Volume', axis=1, level=0).iloc[:, 0]

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    return nifty_df.dropna()

combined_data = load_1min_institutional_data()

if combined_data is None or len(combined_data) == 0:
    st.error("High frequency 1-minute data stream offline or unavailable.")
else:
    # Convert to pure linear arrays
    n_high = combined_data['High_nifty'].to_numpy(dtype=float)
    n_low = combined_data['Low_nifty'].to_numpy(dtype=float)
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    n_vol = combined_data['Volume_nifty'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    raw_timestamps = combined_data.index
    timestamps_formatted = raw_timestamps.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. RUN PURE KALMAN FILTER BASICS (K=0.001 | 0.001x Matrix)
    # Note: 1-min data has internal daily continuous stream, gap tracking applied at day changes
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

    # 3. HIGH-SPEED DYNAMIC VWAP CALCULATION
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

    # 4 & 5. INSTITUTIONAL 3:20 PM VOLUME SURGE & DIRECTION MATCHING ENGINE
    nifty_hints = []
    
    # Pre-identify daily boundaries for volume indexing
    for t in range(num_steps):
        current_time = raw_timestamps[t]
        hour = current_time.hour
        minute = current_time.minute
        
        # Extract slices belonging to current day up to 3:00 PM for base baseline
        current_day = parsed_dates[t]
        day_indices = np.where(parsed_dates == current_day)[0]
        day_indices_before_3pm = [idx for idx in day_indices if raw_timestamps[idx].hour < 15 and idx <= t]
        
        if len(day_indices_before_3pm) > 0:
            avg_base_vol = np.mean(n_vol[day_indices_before_3pm])
        else:
            avg_base_vol = 1.0

        # CRITICAL EVALUATION CLOCK: 3:20 PM Execution Window
        if hour == 15 and minute == 20:
            # Check last 10 minutes average volume surge
            recent_vol_avg = np.mean(n_vol[max(0, t-10):t+1])
            is_institutional_heavy = recent_vol_avg > (avg_base_vol * 2.0) # 2x volume baseline clear breakout
            
            # Pure Structural Rule Alignment
            if n_close[t] > mid_real_line[t] and n_close[t] > vwap[t] and is_institutional_heavy:
                hint = "🟢 INSTITUTIONAL GAP-UP BUILD (BTST)"
            elif n_close[t] < mid_real_line[t] and n_close[t] < vwap[t] and is_institutional_heavy:
                hint = "🔴 INSTITUTIONAL GAP-DOWN BUILD (STBT)"
            else:
                hint = "⏳ WEAK FLOW: SQUARE OFF / NO CARRY"
        
        # 3:21 PM se 3:30 PM tak state ko hold rakhte hain clear visibility ke liye
        elif hour == 15 and minute > 20:
            hint = nifty_hints[-1] if len(nifty_hints) > 0 else "⏳ ANALYZING FLOW"
        else:
            hint = "⏳ INTRADAY TRACKING"
            
        nifty_hints.append(hint)

    # 6 & 7. DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        '1-Min Close': [f"{x:.2f}" for x in n_close],
        'Dynamic VWAP': [f"{x:.2f}" for x in vwap],
        'Kalman Center Line': [f"{x:.2f}" for x in mid_real_line],
        'Minute Volume': [f"{int(x)}" for x in n_vol],
        '📈 INSTITUTIONAL HINT': nifty_hints
    }, index=timestamps_formatted)

    df_reversed = df_table.iloc[::-1]

    def style_institutional_flow(val):
        if "GAP-UP" in str(val):
            return "background-color: #1b5e20; color: white; font-weight: bold; border: 2px solid green;"
        elif "GAP-DOWN" in str(val):
            return "background-color: #b71c1c; color: white; font-weight: bold; border: 2px solid red;"
        elif "WEAK" in str(val):
            return "background-color: #e65100; color: white; font-weight: bold;"
        return ""

    styled_final_df = df_reversed.style.map(style_institutional_flow, subset=['📈 INSTITUTIONAL HINT'])

    # 8. RENDER LIVE MATRIX VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("HFT Institutional Flow Engine operational at 1-Min scale. 3:20 PM prediction algorithm locked!")
