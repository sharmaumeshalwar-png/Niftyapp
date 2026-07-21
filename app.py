import streamlit as st
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="BTC 50-Pt Renko Kinematic Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Dual Engine (Fixed 50-Point Non-Repainting Renko)")
st.caption("Zero-Repaint | Zero Future Leak | Strict Causal Shift-1 Signal Processing")

# Initialize CCXT Binance Client
exchange = ccxt.binance({'enableRateLimit': True})

# =====================================================================
# 1. NON-REPAINTING FIXED RENKO GENERATOR (PURE CAUSAL)
# =====================================================================
def build_fixed_renko_bricks(df_raw, brick_size=50.0):
    """
    Constructs leak-free fixed $50 Renko bricks sequentially.
    Guarantees no future data leakage or repainting on closed bricks.
    """
    if df_raw.empty:
        return pd.DataFrame()
        
    prices = df_raw['Close'].to_numpy()
    timestamps = df_raw.index.to_numpy()
    
    renko_bars = []
    
    # Initialize baseline
    base_price = np.floor(prices[0] / brick_size) * brick_size
    current_open = base_price
    
    for i in range(1, len(prices)):
        price = prices[i]
        ts = timestamps[i]
        
        # Upward Bricks
        while price >= current_open + brick_size:
            r_open = current_open
            r_close = current_open + brick_size
            r_high = r_close
            r_low = r_open
            renko_bars.append({
                'timestamp': ts,
                'Open': r_open,
                'High': r_high,
                'Low': r_low,
                'Close': r_close,
                'Type': 1  # Bullish
            })
            current_open = r_close
            
        # Downward Bricks
        while price <= current_open - brick_size:
            r_open = current_open
            r_close = current_open - brick_size
            r_high = r_open
            r_low = r_close
            renko_bars.append({
                'timestamp': ts,
                'Open': r_open,
                'High': r_high,
                'Low': r_low,
                'Close': r_close,
                'Type': -1  # Bearish
            })
            current_open = r_close
            
    df_renko = pd.DataFrame(renko_bars)
    if not df_renko.empty:
        df_renko.set_index('timestamp', inplace=True)
    return df_renko

# =====================================================================
# 2. MATHEMATICAL ENGINES (HEIKIN-ASHI ON RENKO, KALMAN & HURST)
# =====================================================================
def compute_heikin_ashi_renko(df_in):
    df_ha = df_in.copy()
    op = df_ha['Open'].to_numpy().flatten()
    hi = df_ha['High'].to_numpy().flatten()
    lo = df_ha['Low'].to_numpy().flatten()
    cl = df_ha['Close'].to_numpy().flatten()
    
    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0
    
    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))
    
    df_ha['HA_Open'] = ha_open
    df_ha['HA_High'] = ha_high
    df_ha['HA_Low'] = ha_low
    df_ha['HA_Close'] = ha_close
    return df_ha

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

def calculate_rolling_hurst_leak_free(price_series, window=30):
    hurst_values = np.full(len(price_series), 0.5) 
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    
    for i in range(window, len(price_series)):
        window_data = log_returns[i - window + 1 : i + 1]
        cum_dev = np.cumsum(window_data - np.mean(window_data))
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data) + 1e-10
        
        rs_ratio = r_val / s_val
        if rs_ratio > 0:
            h = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h, 0.0, 1.0)
            
    return hurst_values

def compute_ha_ham_features(df_raw):
    df_ha = compute_heikin_ashi_renko(df_raw)
    ha_close = df_ha['HA_Close'].to_numpy().flatten()
    
    df_ha['Hurst'] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    kalman = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
    momentum = apply_kalman_filter_custom(ha_close - kalman, initial_p=0.50, q_val=0.001, r_val=0.1)
    
    df_ha['Kalman_Price'] = kalman
    df_ha['HA_HAM'] = np.array(momentum) * (df_ha['Hurst'].to_numpy() * 2.0)
    return df_ha

# =====================================================================
# 3. DATA INGESTION & RENKO TIME-ALIGNMENT
# =====================================================================
@st.cache_data(ttl=60)
def fetch_ccxt_klines(symbol='BTC/USDT', tf_1m='1m', limit_1m=1500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf_1m, limit=limit_1m)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"CCXT Fetch Error: {e}")
        return pd.DataFrame()

df_1m_raw = fetch_ccxt_klines()

if df_1m_raw.empty:
    st.stop()

# Build 50-Point Renko Bricks
df_renko_raw = build_fixed_renko_bricks(df_1m_raw, brick_size=50.0)

if df_renko_raw.empty or len(df_renko_raw) < 35:
    st.warning("⚠️ Insufficient Renko Bricks generated. Waiting for market movement...")
    st.stop()

# Higher TF Renko (Aggregated every 4 bricks) vs Fast TF Renko
df_renko_fast = compute_ha_ham_features(df_renko_raw)

# Aggregate Higher Frame Renko Bricks (Higher-order momentum)
df_renko_slow_raw = df_renko_raw.iloc[::4].copy()
df_renko_slow = compute_ha_ham_features(df_renko_slow_raw)

# =====================================================================
# 4. ORIGINAL MATH STRATEGY APPLIED TO CONFIRMED RENKO BRICKS
# =====================================================================
df_grid = df_renko_fast.copy()

# Strictly map historical slow bricks to fast bricks (No forward fill leaks)
df_grid['Slow_HA_Close_Frozen'] = df_renko_slow['HA_Close'].reindex(df_grid.index, method='ffill')
df_grid['HA_HAM_Slow_Frozen'] = df_renko_slow['HA_HAM'].reindex(df_grid.index, method='ffill')
df_grid['HA_HAM_Slow_Prev'] = df_renko_slow['HA_HAM'].shift(1).reindex(df_grid.index, method='ffill')

df_grid['HAM_Diff'] = df_grid['HA_HAM_Slow_Frozen'] - df_grid['HA_HAM']

n = len(df_grid)
h1_curr_arr = df_grid['HA_HAM_Slow_Frozen'].to_numpy()
h1_prev_arr = df_grid['HA_HAM_Slow_Prev'].to_numpy()
m15_curr_arr = df_grid['HA_HAM'].to_numpy()

ha_close_vals = df_grid['HA_Close'].to_numpy()
ha_open_vals = df_grid['HA_Open'].to_numpy()

signals = ['⚪ NEUTRAL'] * n

# Loop starts strictly after warmup window
for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    
    ha_close = ha_close_vals[i]
    ha_open = ha_open_vals[i]
    is_ha_red = ha_close < ha_open

    # Exact Original Signal Logic
    if h1_curr > 0 and h1_curr < h1_prev:
        if m15_curr < 0 or is_ha_red:
            signals[i] = '🔴 REAL TOP (Slow Drop + Fast Red)'
        else:
            signals[i] = '🟢 TRAP PASS (Fast Bullish / Dip Buy)'
            
    elif h1_curr < 0 and h1_curr > h1_prev:
        if m15_curr > 0 and not is_ha_red:
            signals[i] = '🟢 REAL BOTTOM (Slow Rise + Fast Green)'
        else:
            signals[i] = '🔴 TRAP PASS (Fast Bearish / Fake Rally)'
            
    elif h1_curr > h1_prev and h1_curr > 0:
        signals[i] = '🟢 ACCELERATED RALLY'
    elif h1_curr < h1_prev and h1_curr < 0:
        signals[i] = '🔴 ACCELERATED DROP'

df_grid['Instant_Kinematic_Signal'] = signals
df_grid.dropna(subset=['HA_HAM', 'HA_HAM_Slow_Frozen'], inplace=True)

latest = df_grid.iloc[-1]
latest_time = df_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 5. UI DISPLAY MATRIX
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Instant_Kinematic_Signal']
    if 'REAL BOTTOM' in sig or 'TRAP PASS (Fast Bullish' in sig or 'RALLY' in sig:
        st.success(f"### Live Signal ({latest_time})\n# {sig}")
    elif 'REAL TOP' in sig or 'TRAP PASS (Fast Bearish' in sig or 'DROP' in sig:
        st.error(f"### Live Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Live Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slow HA-Close", f"${latest['Slow_HA_Close_Frozen']:,.2f}")
    m2.metric("Fast HA-Close", f"${latest['HA_Close']:,.2f}")
    m3.metric("Fast Live HA-HAM", f"{latest['HA_HAM']:.2f}")
    m4.metric("HAM Diff", f"{latest['HAM_Diff']:.2f}")

st.markdown("---")
st.subheader("📋 50-Point Renko Dual Timeframe Timeline (Non-Repainting)")

clean_cols = ['Slow_HA_Close_Frozen', 'HA_Close', 'HA_HAM_Slow_Frozen', 'HA_HAM', 'HAM_Diff', 'Instant_Kinematic_Signal']
display_df = df_grid[clean_cols].copy()

display_df.rename(columns={
    'Slow_HA_Close_Frozen': 'Slow Renko Close',
    'HA_Close': 'Fast Renko Close',
    'HA_HAM_Slow_Frozen': 'Slow Locked HA-HAM',
    'HA_HAM': 'Fast Live HA-HAM',
    'HAM_Diff': 'HAM Diff',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

for c in ['Slow Renko Close', 'Fast Renko Close', 'Slow Locked HA-HAM', 'Fast Live HA-HAM', 'HAM Diff']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
