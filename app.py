import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("🔥 Nifty & India VIX Super-Sonic Fast Flip Dashboard")

# --- HYPER-DRIVE CONTROLLER SIDEBAR ---
st.sidebar.header("🚀 Hyper-Drive Tuning")
st.sidebar.write("🔴 WARNING: High speed means instant flips on minor movements!")

# Process Noise Slider (Bada number = Instant reaction)
q_value = st.sidebar.slider("Filter Reaction Speed (Q)", min_value=0.5, max_value=50.0, value=15.0, step=0.5)

# Tight Band Multiplier (0.990 aur usse neeche = Super tight/Fastest Flip)
multiplier = st.sidebar.slider("Tight Band Multiplier", min_value=0.950, max_value=1.050, value=0.994, step=0.001)

st.write(f"⚡ **Hyper-Drive Status:** Active | **Q Speed:** {q_value} | **Band Compression:** {multiplier}")

# 1. FUNCTION TO DOWNLOAD AND EXTRACT EXPLICIT MULTIINDEX SERIES
@st.cache_data(ttl=1800) # Reduced TTL for fresher live updates
def load_frozen_data():
    nifty_raw = yf.download('^NSEI', start='2024-07-01', end='2027-01-01', interval='1h')
    vix_raw = yf.download('^INDIAVIX', start='2024-07-01', end='2027-01-01', interval='1h')
    
    if nifty_raw.empty or vix_raw.empty:
        return None

    # Safe MultiIndex Extraction using Cross-Section (.xs)
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw.xs('High', axis=1, level=0).iloc[:, 0]
    nifty_df['Low_nifty'] = nifty_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    nifty_df['Open_nifty'] = nifty_raw.xs('Open', axis=1, level=0).iloc[:, 0]
    nifty_df['Close_nifty'] = nifty_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw.xs('High', axis=1, level=0).iloc[:, 0]
    vix_df['Low_vix'] = vix_raw.xs('Low', axis=1, level=0).iloc[:, 0]
    vix_df['Close_vix'] = vix_raw.xs('Close', axis=1, level=0).iloc[:, 0]

    # Timezone clean-up
    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    # Synchronized Merge
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

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
    
    v_high = combined_data['High_vix'].to_numpy(dtype=float)
    v_low = combined_data['Low_vix'].to_numpy(dtype=float)
    v_close = combined_data['Close_vix'].to_numpy(dtype=float)
    
    num_steps = len(combined_data)
    timestamps = combined_data.index.strftime('%Y-%m-%d %H:%M')
    parsed_dates = combined_data.index.date

    # 2. PURE GAP DE-TRENDING ENGINE
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:  
                cumulative_gap += gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 3. NIFTY HIGH KALMAN FILTER (Hyper-Drive Q Setting)
    b_nifty_high = np.zeros(num_steps)
    x_n_high = n_high_adj[0]
    P_nh, Q_nh, R_nh = 1.0, q_value, 1.0
    for t in range(num_steps):
        K = (P_nh + Q_nh) / (P_nh + Q_nh + R_nh)
        x_n_high = x_n_high + K * (n_high_adj[t] - x_n_high)
        P_nh = (1 - K) * (P_nh + Q_nh)
        b_nifty_high[t] = x_n_high

    # 4. NIFTY LOW KALMAN FILTER (Hyper-Drive Q Setting)
    b_nifty_low = np.zeros(num_steps)
    x_n_low = n_low_adj[0]
    P_nl, Q_nl, R_nl = 1.0, q_value, 1.0
    for t in range(num_steps):
        K = (P_nl + Q_nl) / (P_nl + Q_nl + R_nl)
        x_n_low = x_n_low + K * (n_low_adj[t] - x_n_low)
        P_nl = (1 - K) * (P_nl + Q_nl)
        b_nifty_low[t] = x_n_low

    # 5. ULTRA-COMPRESSION MULTI-LOCK ENGINE
    # Directly pushing the channel boundaries to the absolute limit
    nifty_high_real = (b_nifty_high + cumulative_gap) * multiplier
    nifty_low_real = (b_nifty_low + cumulative_gap) * (2.0 - multiplier)

    # INDIA VIX KALMAN FILTERS
    vifty_high = np.zeros(num_steps)
    x_v_high = v_high[0]
    P_vh, Q_vh, R_vh = 1.0, q_value, 1.0
    for t in range(num_steps):
        K = (P_vh + Q_vh) / (P_vh + Q_vh + R_vh)
        x_v_high = x_v_high + K * (v_high[t] - x_v_high)
        P_vh = (1 - K) * (P_vh + Q_vh)
        vifty_high[t] = x_v_high

    vifty_low = np.zeros(num_steps)
    x_v_low = v_low[0]
    P_vl, Q_vl, R_vl = 1.0, q_value, 1.0
    for t in range(num_steps):
        K = (P_vl + Q_vl) / (P_vl + Q_vl + R_vl)
        x_v_low = x_v_low + K * (v_low[t] - x_v_low)
        P_vl = (1 - K) * (P_vl + Q_vl)
        vifty_low[t] = x_v_low

    # 6. INSTANT TREND SIGNALS
    nifty_signals = []
    current_signal = "⏳ INITIALIZING"
    for t in range(num_steps):
        if n_close[t] > nifty_high_real[t]:
            current_signal = "🟢 BUY"
        elif n_close[t] < nifty_low_real[t]:
            current_signal = "🔴 SELL"
        nifty_signals.append(current_signal)

    # 7. INDIA VIX SIGNALS
    vix_signals = []
    for t in range(num_steps):
        if v_close[t] > vifty_high[t]:
            vix_signals.append("🔴 RISK HIGH")
        elif v_close[t] < vifty_low[t]:
            vix_signals.append("🟢 RISK LOW")
        else:
            vix_signals.append("⚪ SIDEWAYS")

    # DATAFRAME COMPILATION
    df_table = pd.DataFrame({
        'Nifty Close': [f"{x:.2f}" for x in n_close],
        'Nifty High K': [f"{x:.2f}" for x in nifty_high_real],
        'Nifty Low K': [f"{x:.2f}" for x in nifty_low_real],
        '📈 NIFTY HINT': nifty_signals,
        'VIX Close': [f"{x:.2f}" for x in v_close],
        'VIX High K': [f"{x:.2f}" for x in vifty_high],
        'VIX Low K': [f"{x:.2f}" for x in vifty_low],
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

    # 8. RENDER SECURE VIEW
    st.dataframe(styled_final_df, use_container_width=True)
    st.success("Hyper-Drive Active! Zero-Lag Signals are now fully operational.")
