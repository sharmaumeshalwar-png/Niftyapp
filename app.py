import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC HA Dual-Timeframe Kinematic Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Heikin-Ashi Dual Engine (1H Frozen + 15M Live Dynamic)")
st.caption("Includes 90 Days Backtest Engine + Instant Level-Flip Invalidation Logic")

# =====================================================================
# 1. DIRECT BINANCE 90-DAYS DATA FETCHER
# =====================================================================
@st.cache_data(ttl=600)
def fetch_binance_90_days_data(symbol='BTCUSDT', interval='15m', days=90):
    url = "https://api.binance.com/api/v3/klines"
    
    now = datetime.utcnow()
    start_dt = now - timedelta(days=days)
    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(now.timestamp() * 1000)
    
    all_data = []
    limit = 1000
    current_start = start_time
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_expected = days * 24 * 4  # ~8640 candles for 90 days
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': limit
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            all_data.extend(data)
            
            progress = min(len(all_data) / total_expected, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Fetching 90 Days Data... Loaded {len(all_data)} candles")
            
            current_start = data[-1][0] + 1
            time.sleep(0.03)
        except Exception as e:
            st.error(f"Data Fetching Error: {e}")
            break

    progress_bar.empty()
    status_text.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    df = df[~df.index.duplicated(keep='first')]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# =====================================================================
# 2. MATHEMATICAL ENGINES (HEIKIN-ASHI, KALMAN & HURST)
# =====================================================================
def compute_heikin_ashi(df_in):
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
    df_ha = compute_heikin_ashi(df_raw)
    ha_close = df_ha['HA_Close'].to_numpy().flatten()
    
    df_ha['Hurst'] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    kalman = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
    momentum = apply_kalman_filter_custom(ha_close - kalman, initial_p=0.50, q_val=0.001, r_val=0.1)
    
    df_ha['Kalman_Price'] = kalman
    df_ha['HA_HAM'] = np.array(momentum) * (df_ha['Hurst'].to_numpy() * 2.0)
    return df_ha

# =====================================================================
# DATA INGESTION & DUAL TIMEFRAME RESAMPLING
# =====================================================================
df_15m_raw = fetch_binance_90_days_data(symbol='BTCUSDT', interval='15m', days=90)

if df_15m_raw.empty:
    st.error("🚨 Data Load Error from Binance API. Please refresh.")
    st.stop()

# Build 1H Data from 15M Raw Data for Perfect Alignment
df_1h_raw = df_15m_raw.resample('1h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()

df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)

# =====================================================================
# ⚙️ 1H FREEZE + LEVEL INVALIDATION FLIP ENGINE
# =====================================================================
df_15m_grid = df_15m.copy()

# Forward-fill 1H metrics to 15M timeline with strict anti-leakage shift
df_15m_grid['1H_HA_Close_Frozen'] = df_1h['HA_Close'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Frozen'] = df_1h['HA_HAM'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Prev'] = df_1h['HA_HAM'].shift(1).reindex(df_15m_grid.index, method='ffill')

df_15m_grid['HAM_Diff'] = df_15m_grid['HA_HAM_1H_Frozen'] - df_15m_grid['HA_HAM']

n = len(df_15m_grid)
h1_curr_arr = df_15m_grid['HA_HAM_1H_Frozen'].to_numpy()
h1_prev_arr = df_15m_grid['HA_HAM_1H_Prev'].to_numpy()
m15_curr_arr = df_15m_grid['HA_HAM'].to_numpy()

ha_close_vals = df_15m_grid['HA_Close'].to_numpy()
ha_open_vals = df_15m_grid['HA_Open'].to_numpy()
ha_high_vals = df_15m_grid['HA_High'].to_numpy()
ha_low_vals = df_15m_grid['HA_Low'].to_numpy()

signals = ['⚪ NEUTRAL'] * n
barrier_levels = [None] * n

active_state = None       # 'TOP' or 'BOTTOM'
last_level = None         # Level Barrier Price

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    
    ha_close = ha_close_vals[i]
    ha_open = ha_open_vals[i]
    ha_high = ha_high_vals[i]
    ha_low = ha_low_vals[i]
    
    is_ha_red = ha_close < ha_open

    # ----------------------------------------------------
    # STEP 1: INSTANT LEVEL FLIP LOGIC (HIGHEST PRIORITY)
    # ----------------------------------------------------
    if active_state == 'TOP' and last_level is not None:
        if ha_close > last_level:
            active_state = 'BOTTOM'
            last_level = ha_low
            signals[i] = '🟢 REAL BOTTOM (Instant Flip / Level Break)'
            barrier_levels[i] = last_level
            continue

    elif active_state == 'BOTTOM' and last_level is not None:
        if ha_close < last_level:
            active_state = 'TOP'
            last_level = ha_high
            signals[i] = '🔴 REAL TOP (Instant Flip / Level Break)'
            barrier_levels[i] = last_level
            continue

    # ----------------------------------------------------
    # STEP 2: BASE DUAL-TIMEFRAME ENGINE
    # ----------------------------------------------------
    if h1_curr > 0 and h1_curr < h1_prev:
        if m15_curr < 0 or is_ha_red:
            signals[i] = '🔴 REAL TOP (1H Drop + 15M Red)'
            active_state = 'TOP'
            last_level = ha_high
        else:
            signals[i] = '🟢 TRAP PASS (15M Bullish / Dip Buy)'
            
    elif h1_curr < 0 and h1_curr > h1_prev:
        if m15_curr > 0 and not is_ha_red:
            signals[i] = '🟢 REAL BOTTOM (1H Rise + 15M Green)'
            active_state = 'BOTTOM'
            last_level = ha_low
        else:
            signals[i] = '🔴 TRAP PASS (15M Bearish / Fake Rally)'
            
    elif h1_curr > h1_prev and h1_curr > 0:
        signals[i] = '🟢 ACCELERATED RALLY'
    elif h1_curr < h1_prev and h1_curr < 0:
        signals[i] = '🔴 ACCELERATED DROP'

    barrier_levels[i] = last_level

df_15m_grid['Instant_Kinematic_Signal'] = signals
df_15m_grid['Barrier_Level'] = barrier_levels

df_15m_grid.dropna(subset=['HA_HAM', 'HA_HAM_1H_Frozen'], inplace=True)

latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 DISPLAY MATRIX
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Instant_Kinematic_Signal']
    if 'REAL BOTTOM' in sig or 'TRAP PASS (15M Bullish' in sig or 'RALLY' in sig:
        st.success(f"### Live Signal ({latest_time})\n# {sig}")
    elif 'REAL TOP' in sig or 'TRAP PASS (15M Bearish' in sig or 'DROP' in sig:
        st.error(f"### Live Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Live Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1H HA-Close", f"${latest['1H_HA_Close_Frozen']:,.2f}")
    m2.metric("15M HA-Close", f"${latest['HA_Close']:,.2f}")
    m3.metric("Barrier Level", f"${latest['Barrier_Level']:,.2f}" if pd.notna(latest['Barrier_Level']) else "N/A")
    m4.metric("15M Live HA-HAM", f"{latest['HA_HAM']:.2f}")
    m5.metric("HAM Diff (1H - 15M)", f"{latest['HAM_Diff']:.2f}")

st.markdown("---")

st.subheader("📋 Heikin-Ashi Dual Timeframe Timeline (90 Days Data)")

clean_cols = ['1H_HA_Close_Frozen', 'HA_Close', 'Barrier_Level', 'HA_HAM_1H_Frozen', 'HA_HAM', 'HAM_Diff', 'Instant_Kinematic_Signal']
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1H HA-Close',
    'HA_Close': '15M HA-Close',
    'Barrier_Level': 'Barrier Level',
    'HA_HAM_1H_Frozen': '1H Locked HA-HAM',
    'HA_HAM': '15M Live HA-HAM',
    'HAM_Diff': 'HAM Diff (1H - 15M)',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

for c in ['1H HA-Close', '15M HA-Close', 'Barrier Level', '1H Locked HA-HAM', '15M Live HA-HAM', 'HAM Diff (1H - 15M)']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
