import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty Fast-Track 4x Matrix Dashboard")

st.write("Fixed Architecture: Hyper-Fast Gain Active (K=0.02) | 4x Spread Matrix | High Reactivity Mode")

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
    timestamps = combined_data.index.strftime('%Y-%m-%d %H:%M')
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

    # 3. HIGH ACCELERATED FILTER (K set to 0.02)
    b_nifty_high = np.zeros(num_steps)
    b_nifty_high[0] = n_high_adj[0]
    K_hyper = 0.02  # Speed badha kar market swings track karne ke liye update kiya
    
    for t in range(1, num_steps):
        b_nifty_high[t] = b_nifty_high[t-1] + K_hyper * (n_high_adj[t] - b_nifty_high[t-1])

    # 4. LOW ACCELERATED FILTER (K set to 0.02)
    b_nifty_low = np.zeros(num_steps)
    b_nifty_low[0] = n_low_adj[0]
    
    for t in range(1, num_steps):
        b_nifty_low[t] = b_nifty_low[t-1] + K_hyper * (n_low_adj[t] - b_nifty_low[t-1])

    # 5. APPLY 4x SPREAD MULTIPLIER ON ACCELERATED BANDS
    fixed_mid = (b_nifty_high + b_nifty_low) / 2.0
    fixed_spread = b_nifty_high - b_nifty_low
    
    b_nifty_high_4x = fixed_mid + (fixed_spread * 2.0) # 4x Matrix logic intact
    b_nifty_low_4x = fixed_mid - (fixed_spread * 2.0)

    # DYNAMIC STEP REALIGNMENT (With Gaps Re-applied)
    nifty_high_real = b_nifty_high_4x + historical_gaps
    nifty_low_real = b_nifty_low_4x + historical_gaps

    # 6. EXACT HIGH MINUS LOW CALCULATION
    high_minus_low = nifty_high_real - nifty_low_real

    # TREND SIGNALS LOOP
    nifty_signals = []
    current_signal = "⏳ INITIALIZING"
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            current_signal = "🟢 BUY"
        elif n_close[t] < nifty_low_real[t]:
            current_signal = "🔴 SELL"
        nifty_signals.append(current_signal)

    # 7. DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty High K (4x)': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K (4x)': [f"{x:.2f}" for x in nifty_low_real],
        'High - Low': [f"{x:.2f}" for x in high_minus_low],
        '📈 NIFTY HINT': nifty_signals
    }, index=timestamps)

    df_reversed = df_table.iloc[::-1]

    def style_nifty_strict(val):
        if "BUY" in str(val):
            return "background-color: #2e7d32; color: white; font-weight: bold;"
        elif "SELL" in str(val):
            return "background-color: #c62828; color: white; font-weight: bold;"
        return ""

    styled_final_df = df_reversed.style.map(style_nifty_strict, subset=['📈 NIFTY HINT'])

    # 8. RENDER SECURE VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Hyper-Fast K=0.02 with 4x Channels deployed perfectly!")
