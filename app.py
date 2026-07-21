import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Page Configuration
st.set_page_config(page_title="BTC HA Dynamic Engine + Futures Bid/Ask", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Heikin-Ashi Dual Engine (1H Frozen + 15M Dynamic) + Live Futures Order Book")
st.caption("Includes Real-Time BTC Perpetual Futures Bid/Ask Depth + Level Invalidation Logic")

# =====================================================================
# 🌐 LIVE BTC FUTURES BID / ASK FETCHING ENGINE (Binance Public API)
# =====================================================================
def fetch_btc_futures_bid_ask():
    """
    Fetches Live BTCUSDT Perpetual Futures Order Book Depth (Bid / Ask Prices and Quantities)
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/depth"
        params = {"symbol": "BTCUSDT", "limit": 5}
        response = requests.get(url, timeout=3)
        data = response.json()
        
        bids = data.get('bids', []) # [[price, qty], ...]
        asks = data.get('asks', []) # [[price, qty], ...]
        
        df_bids = pd.DataFrame(bids, columns=['Bid Price', 'Bid Size (BTC)']).astype(float)
        df_asks = pd.DataFrame(asks, columns=['Ask Price', 'Ask Size (BTC)']).astype(float)
        
        # Calculate Bid-Ask Spread
        best_bid = df_bids['Bid Price'].iloc[0] if len(df_bids) > 0 else 0
        best_ask = df_asks['Ask Price'].iloc[0] if len(df_asks) > 0 else 0
        spread = best_ask - best_bid
        
        return df_bids, df_asks, best_bid, best_ask, spread
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

# =====================================================================
# MATHEMATICAL ENGINES (HEIKIN-ASHI & KALMAN-HAM)
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
# DATA INGESTION
# =====================================================================
@st.cache_data(ttl=60)
def load_market_data():
    df_1h_raw = yf.download(tickers="BTC-USD", period="60d", interval="1h", progress=False)
    df_15m_raw = yf.download(tickers="BTC-USD", period="14d", interval="15m", progress=False)
    
    for d in [df_1h_raw, df_15m_raw]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if d.index.tz is None:
            d.index = d.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            d.index = d.index.tz_convert('Asia/Kolkata')
            
    return df_1h_raw, df_15m_raw

with st.spinner("Fetching Live 1-Hour, 15-Minute & Futures Bid/Ask Data..."):
    try:
        df_1h_raw, df_15m_raw = load_market_data()
        df_bids, df_asks, best_bid, best_ask, spread = fetch_btc_futures_bid_ask()
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Compute Heikin-Ashi & HAM Features
df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)

# =====================================================================
# ⚙️ 1H FREEZE + 15M STEPWISE ALIGNMENT & INVALIDATION ENGINE
# =====================================================================
df_15m_grid = df_15m.copy()

# Forward Fill 1-Hour HA-Close & HA-HAM on 15M timestamps
df_15m_grid['1H_HA_Close_Frozen'] = df_1h['HA_Close'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Frozen'] = df_1h['HA_HAM'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Prev'] = df_1h['HA_HAM'].shift(1).reindex(df_15m_grid.index, method='ffill')

# HAM Difference
df_15m_grid['HAM_Diff'] = df_15m_grid['HA_HAM_1H_Frozen'] - df_15m_grid['HA_HAM']

# Delta Momentum
df_15m_grid['HA_Close_Diff_15M'] = df_15m_grid['HA_Close'] - df_15m_grid['HA_Close'].shift(1)
df_15m_grid['15M_Delta_Momentum'] = df_15m_grid['HA_Close_Diff_15M'] * df_15m_grid['HA_HAM']

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

active_state = None  # Tracks 'TOP' or 'BOTTOM'
last_level = None    # Level to watch for invalidation

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    
    ha_close = ha_close_vals[i]
    ha_open = ha_open_vals[i]
    ha_high = ha_high_vals[i]
    ha_low = ha_low_vals[i]
    
    is_ha_red = ha_close < ha_open

    # Base Strategy Signal Evaluation
    base_signal = '⚪ NEUTRAL'
    
    if h1_curr > 0 and h1_curr < h1_prev:
        if m15_curr < 0 or is_ha_red:
            base_signal = '🔴 REAL TOP (1H Drop + 15M Red)'
        else:
            base_signal = '🟢 TRAP PASS (15M Bullish / Dip Buy)'
            
    elif h1_curr < 0 and h1_curr > h1_prev:
        if m15_curr > 0 and not is_ha_red:
            base_signal = '🟢 REAL BOTTOM (1H Rise + 15M Green)'
        else:
            base_signal = '🔴 TRAP PASS (15M Bearish / Fake Rally)'
            
    elif h1_curr > h1_prev and h1_curr > 0:
        base_signal = '🟢 ACCELERATED RALLY'
    elif h1_curr < h1_prev and h1_curr < 0:
        base_signal = '🔴 ACCELERATED DROP'

    # State & Invalidation Flip Logic
    if 'REAL TOP' in base_signal:
        active_state = 'TOP'
        last_level = ha_high
        signals[i] = base_signal

    elif 'REAL BOTTOM' in base_signal:
        active_state = 'BOTTOM'
        last_level = ha_low
        signals[i] = base_signal

    elif active_state == 'TOP' and last_level is not None and ha_close > last_level:
        active_state = 'BOTTOM'
        last_level = ha_low
        signals[i] = '🟢 REAL BOTTOM (Level Breakout Flip)'

    elif active_state == 'BOTTOM' and last_level is not None and ha_close < last_level:
        active_state = 'TOP'
        last_level = ha_high
        signals[i] = '🔴 REAL TOP (Level Breakdown Flip)'

    else:
        signals[i] = base_signal

    barrier_levels[i] = last_level

df_15m_grid['Instant_Kinematic_Signal'] = signals
df_15m_grid['Barrier_Level'] = barrier_levels

df_15m_grid.dropna(subset=['HA_HAM', 'HA_HAM_1H_Frozen', '15M_Delta_Momentum'], inplace=True)

latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 DISPLAY MATRIX & LIVE FUTURES BID / ASK BOOK
# =====================================================================
st.markdown("---")

# Metrics Display
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("1H HA-Close", f"${latest['1H_HA_Close_Frozen']:,.2f}")
m2.metric("15M HA-Close", f"${latest['HA_Close']:,.2f}")
m3.metric("Futures Best Bid", f"${best_bid:,.2f}" if best_bid else "N/A")
m4.metric("Futures Best Ask", f"${best_ask:,.2f}" if best_ask else "N/A")
m5.metric("Bid-Ask Spread", f"${spread:.2f}")
m6.metric("Active Barrier Level", f"${latest['Barrier_Level']:,.2f}" if latest['Barrier_Level'] else "N/A")

st.markdown("---")

col_sig, col_depth = st.columns([1.2, 1])

with col_sig:
    sig = latest['Instant_Kinematic_Signal']
    if 'REAL BOTTOM' in sig or 'TRAP PASS (15M Bullish' in sig or 'RALLY' in sig:
        st.success(f"### Live Signal ({latest_time})\n# {sig}")
    elif 'REAL TOP' in sig or 'TRAP PASS (15M Bearish' in sig or 'DROP' in sig:
        st.error(f"### Live Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Live Signal ({latest_time})\n# {sig}")

with col_depth:
    st.subheader("📖 BTC Perpetual Futures Order Book (Depth Top 5)")
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**🟢 Bids (Buy Orders)**")
        st.dataframe(df_bids, use_container_width=True, height=200)
    with col_a:
        st.markdown("**🔴 Asks (Sell Orders)**")
        st.dataframe(df_asks, use_container_width=True, height=200)

st.markdown("---")

# Full Timeline Dataframe
st.subheader("📋 Heikin-Ashi Dual Timeframe Timeline")

clean_cols = ['1H_HA_Close_Frozen', 'HA_Close', 'Barrier_Level', 'HA_HAM_1H_Frozen', 'HA_HAM', 'HAM_Diff', '15M_Delta_Momentum', 'Instant_Kinematic_Signal']
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1H HA-Close',
    'HA_Close': '15M HA-Close',
    'Barrier_Level': 'Active Barrier Level',
    'HA_HAM_1H_Frozen': '1H Locked HA-HAM',
    'HA_HAM': '15M Live HA-HAM',
    'HAM_Diff': 'HAM Diff (1H - 15M)',
    '15M_Delta_Momentum': '15M Delta Momentum',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

for c in ['1H HA-Close', '15M HA-Close', 'Active Barrier Level', '1H Locked HA-HAM', '15M Live HA-HAM', 'HAM Diff (1H - 15M)', '15M Delta Momentum']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=600)
