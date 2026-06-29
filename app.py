import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Nifty Classic High-Low Pure 1x Dashboard")

st.write("Fixed Architecture: Constant Gain Filter Active (K=0.001) | Pure 1x Scale | Pure Gap Engine Restored")

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

    # 3. HIGH FIXED-GAIN FILTER (Pure 1x - No Multipliers)
    b_nifty_high = np.zeros(num_steps)
    b_nifty_high[0] = n_high_adj[0]
    K_fixed = 0.001
    
    for t in range(1, num_steps):
        b_nifty_high[t] = b_nifty_high[t-1] + K_fixed * (n_high_adj[t] - b_nifty_high[t-1])

    # 4. LOW FIXED-GAIN FILTER (Pure 1x - No Multipliers)
    b_nifty_low = np.zeros(num_steps)
    b_nifty_low[0] = n_low_adj[0]
    
    for t in range(1, num_steps):
        b_nifty_low[t] = b_nifty_low[t-1] + K_fixed * (n_low_adj[t] - b_nifty_low[t-1])

    # 5. DYNAMIC STEP REALIGNMENT (Direct 1x Base Bands + Gaps)
    nifty_high_real = b_nifty_high + historical_gaps
    nifty_low_real = b_nifty_low + historical_gaps

    # 6. FIXED SATEEK TREND SIGNALS (Continuous Green/Red Only)
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
        'Nifty High K (1x)': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K (1x)': [f"{x:.2f}" for x in nifty_low_real],
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
    st.success("Pure 1x Fixed-Gain Engine Restored Successfully!")
