import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Einstein E=mc² Energy Candle (Pure Price)", layout="wide", initial_sidebar_state="collapsed")

st.title("⚛️ Einstein E=mc² Relativistic Energy Candle (Volume-Free Engine)")
st.caption("Pure Price Density & Relativistic Acceleration Mechanics — Independent of Volume Data.")

# =====================================================================
# ⚛️ VOLUME-FREE EINSTEIN RELATIVISTIC CANDLE ENGINE
# =====================================================================
def compute_einstein_energy_candles_pure(df_in):
    df_e = df_in.copy()
    
    op = df_e['Open'].to_numpy().flatten()
    hi = df_e['High'].to_numpy().flatten()
    lo = df_e['Low'].to_numpy().flatten()
    cl = df_e['Close'].to_numpy().flatten()
    
    n = len(df_e)
    
    # 1. Price Velocity (v) & Acceleration (a)
    v = np.zeros(n)
    v[1:] = cl[1:] - cl[:-1]
    
    a = np.zeros(n)
    a[1:] = v[1:] - v[:-1]
    
    # 2. Pure Price Mass Density (m) = Body / Spread Ratio
    spread = np.maximum(hi - lo, 1e-6)
    body = np.abs(cl - op)
    mass = body / spread  # Relativistic Mass Density between 0 and 1
    
    # 3. Einstein Energy Metrics (E = m * v^2 * Sign(Acceleration))
    # Accel direction tells if energy is expanding or contracting
    energy = mass * (v ** 2) * np.sign(np.where(a == 0, v, a))
    
    # 4. Relativistic Space Contraction / Expansion Shift Factor
    energy_abs_mean = np.mean(np.abs(energy)) + 1e-6
    rel_factor = energy / energy_abs_mean
    rel_factor = np.clip(rel_factor, -2.5, 2.5) # Relativistic Space limits
    
    # 5. Compute E-OHLC (Relativistic Energy OHLC)
    e_close = cl + rel_factor * (spread * 0.35)
    e_open = np.zeros(n)
    e_open[0] = (op[0] + cl[0]) / 2.0
    
    for i in range(1, n):
        e_open[i] = (e_open[i-1] + e_close[i-1]) / 2.0
        
    e_high = np.maximum(hi, np.maximum(e_open, e_close))
    e_low = np.minimum(lo, np.minimum(e_open, e_close))
    
    df_e['E_Open'] = e_open
    df_e['E_High'] = e_high
    df_e['E_Low'] = e_low
    df_e['E_Close'] = e_close
    df_e['Energy_Val'] = energy
    
    return df_e

def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p  
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def compute_e_ham_features(df_raw):
    df_e = compute_einstein_energy_candles_pure(df_raw)
    e_close = df_e['E_Close'].to_numpy().flatten()
    
    kalman = apply_kalman_filter_custom(e_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
    momentum = apply_kalman_filter_custom(e_close - kalman, initial_p=0.50, q_val=0.001, r_val=0.1)
    
    df_e['E_HAM'] = np.array(momentum)
    return df_e

# =====================================================================
# DATA INGESTION
# =====================================================================
with st.spinner("Fetching Live BTC-USD Data (Pure Price Mode)..."):
    try:
        df_1h_raw = yf.download(tickers="BTC-USD", period="60d", interval="1h")
        df_15m_raw = yf.download(tickers="BTC-USD", period="14d", interval="15m")
        
        for d in [df_1h_raw, df_15m_raw]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # No Volume check needed
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                d.index = d.index.tz_convert('Asia/Kolkata')
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Compute Pure E-Candle & HAM Features
df_1h = compute_e_ham_features(df_1h_raw)
df_15m = compute_e_ham_features(df_15m_raw)

# =====================================================================
# ⚙️ FREEZE & ALIGNMENT ENGINE
# =====================================================================
df_15m_grid = df_15m.copy()

df_15m_grid['1H_E_Close_Frozen'] = df_1h['E_Close'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['E_HAM_1H_Frozen'] = df_1h['E_HAM'].reindex(df_15m_grid.index, method='ffill')

# Energy HAM Difference
df_15m_grid['E_HAM_Diff'] = df_15m_grid['E_HAM_1H_Frozen'] - df_15m_grid['E_HAM']

n = len(df_15m_grid)
e_close_vals = df_15m_grid['E_Close'].to_numpy()
e_open_vals = df_15m_grid['E_Open'].to_numpy()

signals = ['⚪ NEUTRAL'] * n

for i in range(2, n):
    h1_ham = df_15m_grid['E_HAM_1H_Frozen'].iloc[i]
    m15_ham = df_15m_grid['E_HAM'].iloc[i]
    is_red = e_close_vals[i] < e_open_vals[i]
    
    if h1_ham > 0 and (m15_ham < 0 or is_red):
        signals[i] = '🔴 ENERGY DRAIN (High Mass Reversal)'
    elif h1_ham < 0 and (m15_ham > 0 and not is_red):
        signals[i] = '🟢 ENERGY IGNITION (Quantum Bottom)'
    elif m15_ham > 0:
        signals[i] = '🟢 RELATIVISTIC RALLY'
    elif m15_ham < 0:
        signals[i] = '🔴 RELATIVISTIC DROP'

df_15m_grid['Kinematic_Signal'] = signals
df_15m_grid.dropna(subset=['E_HAM', 'E_HAM_1H_Frozen'], inplace=True)

latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 DISPLAY MATRIX
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Kinematic_Signal']
    if 'IGNITION' in sig or 'RALLY' in sig:
        st.success(f"### Live Energy Signal ({latest_time})\n# {sig}")
    elif 'DRAIN' in sig or 'DROP' in sig:
        st.error(f"### Live Energy Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Live Energy Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1H E-Close", f"${latest['1H_E_Close_Frozen']:,.2f}")
    m2.metric("15M E-Close", f"${latest['E_Close']:,.2f}")
    m3.metric("1H Locked E-HAM", f"{latest['E_HAM_1H_Frozen']:.2f}")
    m4.metric("15M Live E-HAM", f"{latest['E_HAM']:.2f}")
    m5.metric("E-HAM Diff", f"{latest['E_HAM_Diff']:.2f}")

st.markdown("---")

st.subheader("📋 Einstein E=mc² Pure Relativistic Timeline")

clean_cols = ['1H_E_Close_Frozen', 'E_Close', 'E_HAM_1H_Frozen', 'E_HAM', 'E_HAM_Diff', 'Kinematic_Signal']
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_E_Close_Frozen': '1H E-Close',
    'E_Close': '15M E-Close',
    'E_HAM_1H_Frozen': '1H Locked E-HAM',
    'E_HAM': '15M Live E-HAM',
    'E_HAM_Diff': 'E-HAM Diff (1H - 15M)',
    'Kinematic_Signal': 'Relativistic Energy Signal'
}, inplace=True)

for c in ['1H E-Close', '15M E-Close', '1H Locked E-HAM', '15M Live E-HAM', 'E-HAM Diff (1H - 15M)']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
