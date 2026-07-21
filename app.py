import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="BTC Renko Dual-Timeframe Kinematic Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Renko 50-Point Dual Engine (1H Frozen + Renko Live Dynamic)")
st.caption("Includes 90 Days Backtest Engine + Completed 50-Point Renko Bricks Invalidation Logic")

# =====================================================================
# 1. ROBUST MULTI-ENDPOINT BINANCE 90-DAYS DATA FETCHER
# =====================================================================
@st.cache_data(ttl=600)
def fetch_binance_90_days_data(symbol='BTCUSDT', interval='15m', days=90):
    # Multiple public endpoints to bypass IP/Mirror restrictions
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
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
    
    active_url_idx = 0

    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': limit
        }
        
        success = False
        # Retry loop across available Binance mirror endpoints
        for idx in range(len(endpoints)):
            target_url = endpoints[(active_url_idx + idx) % len(endpoints)]
            try:
                res = requests.get(target_url, params=params, headers=headers, timeout=12)
                
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        all_data.extend(data)
                        current_start = data[-1][0] + 1
                        success = True
                        active_url_idx = (active_url_idx + idx) % len(endpoints)
                        break
                elif res.status_code in [403, 451]:
                    # IP restricted / Cloud block detected
                    continue
            except Exception:
                continue

        if not success:
            st.warning("⚠️ High latency or API endpoint blocking detected. Finalizing loaded candles.")
            break

        progress = min(len(all_data) / total_expected, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Fetching Data... Loaded {len(all_data)} candles")
        time.sleep(0.02)

    progress_bar.empty()
    status_text.empty()

    if not all_data:
        st.error("🚨 Connection Failed: Binance API endpoints blocked. If using Streamlit Cloud, hosting server IPs might be blocked by Binance.")
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
# 2. RENKO BRICK BUILDER (STRICT 50-POINT COMPLETED BRICKS ONLY)
# =====================================================================
def build_pure_renko_bricks(df_in, brick_size=50.0):
    closes = df_in['Close'].to_numpy()
    timestamps = df_in.index
    
    renko_data = []
    if len(closes) == 0:
        return pd.DataFrame()

    base_price = np.floor(closes[0] / brick_size) * brick_size
    
    for i in range(len(closes)):
        price = closes[i]
        ts = timestamps[i]
        diff = price - base_price
        
        if diff >= brick_size:
            num_bricks = int(diff // brick_size)
            for _ in range(num_bricks):
                open_p = base_price
                close_p = base_price + brick_size
                renko_data.append({
                    'timestamp': ts,
                    'Open': open_p,
                    'High': close_p,
                    'Low': open_p,
                    'Close': close_p,
                    'Type': 'GREEN'
                })
                base_price = close_p
                
        elif diff <= -brick_size:
            num_bricks = int(abs(diff) // brick_size)
            for _ in range(num_bricks):
                open_p = base_price
                close_p = base_price - brick_size
                renko_data.append({
                    'timestamp': ts,
                    'Open': open_p,
                    'High': open_p,
                    'Low': close_p,
                    'Close': close_p,
                    'Type': 'RED'
                })
                base_price = close_p

    df_renko = pd.DataFrame(renko_data)
    if not df_renko.empty:
        df_renko.set_index('timestamp', inplace=True)
    return df_renko

# =====================================================================
# 3. MATHEMATICAL ENGINES (HEIKIN-ASHI ON RENKO, KALMAN & HURST)
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
# DATA INGESTION & RENKO GENERATION
# =====================================================================
df_15m_raw = fetch_binance_90_days_data(symbol='BTCUSDT', interval='15m', days=90)

if df_15m_raw.empty:
    st.error("🚨 Unable to load data from Binance API endpoints. If you are hosted on Streamlit Cloud, Binance might be blocking cloud server IPs. Try running locally or refreshing.")
    st.stop()

# 1. Generate Completed 50-Point Renko Bricks
df_renko_raw = build_pure_renko_bricks(df_15m_raw, brick_size=50.0)

if df_renko_raw.empty:
    st.error("🚨 Renko Brick Calculation Failure.")
    st.stop()

# Build 1H Data from 15M Raw Data for Higher Timeframe Baseline
df_1h_raw = df_15m_raw.resample('1h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()

df_1h = compute_ha_ham_features(df_1h_raw)
df_renko = compute_ha_ham_features(df_renko_raw)

# =====================================================================
# ⚙️ 1H FREEZE + LEVEL INVALIDATION FLIP ENGINE (ON RENKO BRICKS)
# =====================================================================
df_renko_grid = df_renko.copy()

# Forward-fill 1H metrics to Renko timeline with strict anti-leakage shift
df_renko_grid['1H_HA_Close_Frozen'] = df_1h['HA_Close'].reindex(df_renko_grid.index, method='ffill')
df_renko_grid['HA_HAM_1H_Frozen'] = df_1h['HA_HAM'].reindex(df_renko_grid.index, method='ffill')
df_renko_grid['HA_HAM_1H_Prev'] = df_1h['HA_HAM'].shift(1).reindex(df_renko_grid.index, method='ffill')

df_renko_grid['HAM_Diff'] = df_renko_grid['HA_HAM_1H_Frozen'] - df_renko_grid['HA_HAM']

n = len(df_renko_grid)
h1_curr_arr = df_renko_grid['HA_HAM_1H_Frozen'].to_numpy()
h1_prev_arr = df_renko_grid['HA_HAM_1H_Prev'].to_numpy()
renko_curr_arr = df_renko_grid['HA_HAM'].to_numpy()

ha_close_vals = df_renko_grid['HA_Close'].to_numpy()
ha_open_vals = df_renko_grid['HA_Open'].to_numpy()
ha_high_vals = df_renko_grid['HA_High'].to_numpy()
ha_low_vals = df_renko_grid['HA_Low'].to_numpy()

signals = ['⚪ NEUTRAL'] * n
barrier_levels = [None] * n

active_state = None       # 'TOP' or 'BOTTOM'
last_level = None         # Level Barrier Price

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    renko_curr = renko_curr_arr[i]
    
    ha_close = ha_close_vals[i]
    ha_open = ha_open_vals[i]
    ha_high = ha_high_vals[i]
    ha_low = ha_low_vals[i]
    
    is_renko_red = ha_close < ha_open

    # ----------------------------------------------------
    # STEP 1: INSTANT LEVEL FLIP LOGIC (HIGHEST PRIORITY)
    # ----------------------------------------------------
    if active_state == 'TOP' and last_level is not None:
        if ha_close > last_level:
            active_state = 'BOTTOM'
            last_level = ha_low
            signals[i] = '🟢 REAL BOTTOM (Instant Flip / Renko Level Break)'
            barrier_levels[i] = last_level
            continue

    elif active_state == 'BOTTOM' and last_level is not None:
        if ha_close < last_level:
            active_state = 'TOP'
            last_level = ha_high
            signals[i] = '🔴 REAL TOP (Instant Flip / Renko Level Break)'
            barrier_levels[i] = last_level
            continue

    # ----------------------------------------------------
    # STEP 2: BASE DUAL-TIMEFRAME ENGINE
    # ----------------------------------------------------
    if h1_curr > 0 and h1_curr < h1_prev:
        if renko_curr < 0 or is_renko_red:
            signals[i] = '🔴 REAL TOP (1H Drop + Renko Red)'
            active_state = 'TOP'
            last_level = ha_high
        else:
            signals[i] = '🟢 TRAP PASS (Renko Bullish / Dip Buy)'
            
    elif h1_curr < 0 and h1_curr > h1_prev:
        if renko_curr > 0 and not is_renko_red:
            signals[i] = '🟢 REAL BOTTOM (1H Rise + Renko Green)'
            active_state = 'BOTTOM'
            last_level = ha_low
        else:
            signals[i] = '🔴 TRAP PASS (Renko Bearish / Fake Rally)'
            
    elif h1_curr > h1_prev and h1_curr > 0:
        signals[i] = '🟢 ACCELERATED RALLY'
    elif h1_curr < h1_prev and h1_curr < 0:
        signals[i] = '🔴 ACCELERATED DROP'

    barrier_levels[i] = last_level

df_renko_grid['Instant_Kinematic_Signal'] = signals
df_renko_grid['Barrier_Level'] = barrier_levels

df_renko_grid.dropna(subset=['HA_HAM', 'HA_HAM_1H_Frozen'], inplace=True)

latest = df_renko_grid.iloc[-1]
latest_time = df_renko_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 DISPLAY MATRIX
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Instant_Kinematic_Signal']
    if 'REAL BOTTOM' in sig or 'TRAP PASS (Renko Bullish' in sig or 'RALLY' in sig:
        st.success(f"### Live Completed Renko Signal ({latest_time})\n# {sig}")
    elif 'REAL TOP' in sig or 'TRAP PASS (Renko Bearish' in sig or 'DROP' in sig:
        st.error(f"### Live Completed Renko Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Live Completed Renko Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("1H HA-Close", f"${latest['1H_HA_Close_Frozen']:,.2f}")
    m2.metric("Renko HA-Close", f"${latest['HA_Close']:,.2f}")
    m3.metric("Barrier Level", f"${latest['Barrier_Level']:,.2f}" if pd.notna(latest['Barrier_Level']) else "N/A")
    m4.metric("Renko HA-HAM", f"{latest['HA_HAM']:.2f}")
    m5.metric("HAM Diff (1H - Renko)", f"{latest['HAM_Diff']:.2f}")

st.markdown("---")

st.subheader("📋 50-Point Completed Renko Bricks Timeline (90 Days)")

clean_cols = ['1H_HA_Close_Frozen', 'HA_Close', 'Barrier_Level', 'HA_HAM_1H_Frozen', 'HA_HAM', 'HAM_Diff', 'Instant_Kinematic_Signal']
display_df = df_renko_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1H HA-Close',
    'HA_Close': 'Renko HA-Close',
    'Barrier_Level': 'Barrier Level',
    'HA_HAM_1H_Frozen': '1H Locked HA-HAM',
    'HA_HAM': 'Renko Live HA-HAM',
    'HAM_Diff': 'HAM Diff (1H - Renko)',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

for c in ['1H HA-Close', 'Renko HA-Close', 'Barrier Level', '1H Locked HA-HAM', 'Renko Live HA-HAM', 'HAM Diff (1H - Renko)']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
