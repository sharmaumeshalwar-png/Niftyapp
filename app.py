import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty Closing Velocity 0.001 (BTST / STBT Engine)")

st.write("Option Architecture: K=0.001 | 0.001x Matrix | Institutional Closing Velocity Filter Active")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT MULTIINDEX SERIES
@st.cache_data(ttl=3600)
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    
    if nifty_raw.empty:
        return None

    # Safe MultiIndex Extraction using Cross-Section (.xs)
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    # Timezone clean-up
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    nifty_df = nifty_df[~nifty_df.index.duplicated(keep='first')]
    
    return nifty_df.dropna()

# Execute data engine
combined_data = load_frozen_data()

if combined_data is None or len(combined_data) == 0:
    st.error("Data extraction error.")
else:
    # Pure Linear Arrays
    n_high = combined_data['High_nifty'].to_numpy(dtype=float)
    n_low = combined_data['Low_nifty'].to_numpy(dtype=float)
    n_open = combined_data['Open_nifty'].to_numpy(dtype=float)
    n_close = combined_data['Close_nifty'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    raw_timestamps = combined_data.index
    timestamps_formatted = raw_timestamps.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. PURE GAP DE-TRENDING ENGINE (Tracking Matrix)
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

    # 3. HIGH ACTIVE SPEED FILTER (K = 0.001)
    b_nifty_high = np.zeros(num_steps)
    b_nifty_high[0] = n_high_adj[0]
    K_factor = 0.001  
    
    for t in range(1, num_steps):
        b_nifty_high[t] = b_nifty_high[t-1] + K_factor * (n_high_adj[t] - b_nifty_high[t-1])

    # 4. LOW ACTIVE SPEED FILTER
    b_nifty_low = np.zeros(num_steps)
    b_nifty_low[0] = n_low_adj[0]
    
    for t in range(1, num_steps):
        b_nifty_low[t] = b_nifty_low[t-1] + K_factor * (n_low_adj[t] - b_nifty_low[t-1])

    # 5. APPLY ULTRA-COMPRESSED 0.001x SPREAD MULTIPLIER (0.0005x on each side)
    fixed_mid = (b_nifty_high + b_nifty_low) / 2.0
    fixed_spread = b_nifty_high - b_nifty_low
    
    b_nifty_high_0001x = fixed_mid + (fixed_spread * 0.0005) 
    b_nifty_low_0001x = fixed_mid - (fixed_spread * 0.0005)

    # DYNAMIC STEP REALIGNMENT (With Gaps Re-applied)
    nifty_high_real = b_nifty_high_0001x + historical_gaps
    nifty_low_real = b_nifty_low_0001x + historical_gaps
    high_minus_low = nifty_high_real - nifty_low_real
    mid_real_line = (nifty_high_real + nifty_low_real) / 2.0

    # 6 & 7. CLOSING VELOCITY OPTION CRITERIA
    nifty_signals = []
    current_state = "⏳ NO POSITION"

    for t in range(num_steps):
        current_time = raw_timestamps[t]
        hour = current_time.hour
        
        # Calculate instant shift momentum
        if t >= 1:
            instant_slope = mid_real_line[t] - mid_real_line[t-1]
        else:
            instant_slope = 0.0

        # Targeted Check for Closing Hour Candles (2 PM to 3:30 PM execution frames)
        if hour == 14 or hour == 15:
            # Sensitive velocity parameters matching your tight matrix setup
            if n_close[t] >= mid_real_line[t] and instant_slope >= 0.02:
                current_state = "🟢 BTST: CARRY BUY"
            elif n_close[t] < mid_real_line[t] and instant_slope <= -0.02:
                current_state = "🔴 STBT: CARRY SELL"
            else:
                current_state = "⏳ CHOP: NO CARRY"
        else:
            # Persistent mapping to lock intraday state transitions cleanly
            if "BUY" in current_state:
                current_state = "🟢 HOLD OVERNIGHT BUY"
            elif "SELL" in current_state:
                current_state = "🔴 HOLD OVERNIGHT SELL"
            else:
                current_state = "⏳ SCANNING SESSION"

        nifty_signals.append(current_state)

    # 8. DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty High K (0.001x)': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K (0.001x)': [f"{x:.2f}" for x in nifty_low_real],
        'High - Low': [f"{x:.2f}" for x in high_minus_low],
        '📈 NIFTY HINT': nifty_signals
    }, index=timestamps_formatted)

    df_reversed = df_table.iloc[::-1]

    def style_nifty_strict(val):
        if "CARRY BUY" in str(val):
            return "background-color: #1b5e20; color: white; font-weight: bold; border: 2px solid #a5d6a7;"
        elif "CARRY SELL" in str(val):
            return "background-color: #b71c1c; color: white; font-weight: bold; border: 2px solid #ef9a9a;"
        elif "HOLD" in str(val):
            return "color: #b0bec5; font-style: italic;"
        return ""

    styled_final_df = df_reversed.style.map(style_nifty_strict, subset=['📈 NIFTY HINT'])

    # RENDER VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Closing Velocity Dashboard updated. Sensitive Overnight entry parameters deployed!")
