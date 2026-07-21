import streamlit as st
import numpy as np
import pandas as pd
import requests

# Page Setup
st.set_page_config(
    page_title="BTC Kinematic Engine + Real Binance Volume", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.title("⚡ BTC Dual Engine + Real Binance Market Volume Data")
st.caption("Direct Binance Public API Pipeline (100% Real Taker Orderflow - Zero Synthetic Math)")

# =====================================================================
# 🌐 BINANCE OFFICIAL REAL DATA ENGINE (WITH FALLBACK MIRRORS)
# =====================================================================
def fetch_btc_data_binance():
    """
    Fetches 100% REAL BTC Historical Candles & Taker Buy/Sell Volumes from Binance.
    Includes multi-endpoint fallback to bypass ISP/Geo-blocking seamlessly.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    # Binance Base Endpoints (Primary + Official Public Vision Mirror)
    endpoints = [
        "https://data-api.binance.vision/api/v3/klines",  # Public Market Data Mirror (Zero ISP Block)
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]
    
    params_15m = {'symbol': 'BTCUSDT', 'interval': '15m', 'limit': 1000}
    params_1h = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 500}
    
    cols = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Volume', 'Trades', 
        'Taker_Buy_Base_Vol', 'Taker_Buy_Quote_Vol', 'Ignore'
    ]
    
    res_15m_json = None
    res_1h_json = None
    
    for url in endpoints:
        try:
            r15 = requests.get(url, params=params_15m, headers=headers, timeout=5)
            r1 = requests.get(url, params=params_1h, headers=headers, timeout=5)
            
            if r15.status_code == 200 and r1.status_code == 200:
                res_15m_json = r15.json()
                res_1h_json = r1.json()
                break
        except Exception:
            continue
            
    if not res_15m_json or not res_1h_json:
        return pd.DataFrame(), pd.DataFrame()
        
    df_15m = pd.DataFrame(res_15m_json, columns=cols)
    df_1h = pd.DataFrame(res_1h_json, columns=cols)
    
    def process_df(df_raw):
        if df_raw.empty or not isinstance(df_raw, pd.DataFrame):
            return pd.DataFrame()
            
        float_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Taker_Buy_Base_Vol']
        for c in float_cols:
            df_raw[c] = df_raw[c].astype(float)
            
        df_raw['Open Time'] = pd.to_datetime(df_raw['Open Time'], unit='ms', utc=True).dt.tz_convert('Asia/Kolkata')
        df_raw.set_index('Open Time', inplace=True)
        
        # 🟢 100% REAL MARKET TAKER VOLUME SPLIT
        df_raw['Market_Buy_Vol'] = df_raw['Taker_Buy_Base_Vol']                         # Actual Market Buy Orders
        df_raw['Market_Sell_Vol'] = df_raw['Volume'] - df_raw['Taker_Buy_Base_Vol']     # Actual Market Sell Orders
        df_raw['Total_Volume'] = df_raw['Volume']
        df_raw['Candle_Range'] = df_raw['High'] - df_raw['Low']
        
        return df_raw

    return process_df(df_1h), process_df(df_15m)

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
# DATA PIPELINE EXECUTION
# =====================================================================
with st.spinner("Fetching Real Market Data from Binance & Computing Indicators..."):
    df_1h_raw, df_15m_raw = fetch_btc_data_binance()

if df_1h_raw.empty or df_15m_raw.empty:
    st.error("🚨 Failed to fetch Binance market data. Please check network connection.")
    st.stop()

# Compute Features
df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)

# Forward Fill 1-Hour Signals on 15M Grid
df_15m_grid = df_15m.copy()
df_15m_grid['1H_HA_Close_Frozen'] = df_1h['HA_Close'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Frozen'] = df_1h['HA_HAM'].reindex(df_15m_grid.index, method='ffill')
df_15m_grid['HA_HAM_1H_Prev'] = df_1h['HA_HAM'].shift(1).reindex(df_15m_grid.index, method='ffill')

df_15m_grid['HAM_Diff'] = df_15m_grid['HA_HAM_1H_Frozen'] - df_15m_grid['HA_HAM']
df_15m_grid['HA_Close_Diff_15M'] = df_15m_grid['HA_Close'] - df_15m_grid['HA_Close'].shift(1)
df_15m_grid['15M_Delta_Momentum'] = df_15m_grid['HA_Close_Diff_15M'] * df_15m_grid['HA_HAM']

# Kinematic State Engine
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

active_state = None
last_level = None

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    
    ha_close = ha_close_vals[i]
    ha_open = ha_open_vals[i]
    ha_high = ha_high_vals[i]
    ha_low = ha_low_vals[i]
    
    is_ha_red = ha_close < ha_open
    base_signal = '⚪ NEUTRAL'
    
    if h1_curr > 0 and h1_curr < h1_prev:
        base_signal = '🔴 REAL TOP (1H Drop + 15M Red)' if (m15_curr < 0 or is_ha_red) else '🟢 TRAP PASS (15M Bullish)'
    elif h1_curr < 0 and h1_curr > h1_prev:
        base_signal = '🟢 REAL BOTTOM (1H Rise + 15M Green)' if (m15_curr > 0 and not is_ha_red) else '🔴 TRAP PASS (15M Bearish)'
    elif h1_curr > h1_prev and h1_curr > 0:
        base_signal = '🟢 ACCELERATED RALLY'
    elif h1_curr < h1_prev and h1_curr < 0:
        base_signal = '🔴 ACCELERATED DROP'

    # Invalidation Flip Logic
    if 'REAL TOP' in base_signal:
        active_state, last_level, signals[i] = 'TOP', ha_high, base_signal
    elif 'REAL BOTTOM' in base_signal:
        active_state, last_level, signals[i] = 'BOTTOM', ha_low, base_signal
    elif active_state == 'TOP' and last_level is not None and ha_close > last_level:
        active_state, last_level, signals[i] = 'BOTTOM', ha_low, '🟢 REAL BOTTOM (Breakout Flip)'
    elif active_state == 'BOTTOM' and last_level is not None and ha_close < last_level:
        active_state, last_level, signals[i] = 'TOP', ha_high, '🔴 REAL TOP (Breakdown Flip)'
    else:
        signals[i] = base_signal

    barrier_levels[i] = last_level

df_15m_grid['Instant_Kinematic_Signal'] = signals
df_15m_grid['Barrier_Level'] = barrier_levels

df_15m_grid.dropna(subset=['HA_HAM', '15M_Delta_Momentum'], inplace=True)
latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 METRICS & TABLE DISPLAY
# =====================================================================
st.markdown("---")

# Signal Banner
sig = latest['Instant_Kinematic_Signal']
if 'REAL BOTTOM' in sig or 'TRAP PASS (15M Bullish' in sig or 'RALLY' in sig:
    st.success(f"### Live Signal ({latest_time})\n# {sig}")
elif 'REAL TOP' in sig or 'TRAP PASS (15M Bearish' in sig or 'DROP' in sig:
    st.error(f"### Live Signal ({latest_time})\n# {sig}")
else:
    st.warning(f"### Live Signal ({latest_time})\n# {sig}")

st.markdown("---")

# Metrics Display
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("15M HA Close", f"${latest['HA_Close']:,.2f}")
m2.metric("Candle Low (Price)", f"${latest['Low']:,.2f}")
m3.metric("Candle High (Price)", f"${latest['High']:,.2f}")
m4.metric("Candle Range ($)", f"${latest['Candle_Range']:.2f}")
m5.metric("Real Buy Vol (BTC)", f"{latest['Market_Buy_Vol']:,.2f}")
m6.metric("Real Sell Vol (BTC)", f"{latest['Market_Sell_Vol']:,.2f}")

st.markdown("---")

st.subheader("📋 Historical Candle Data (Includes 100% Real Binance Buy/Sell Orderflow)")

# Format Columns for Clean Table
clean_cols = [
    '1H_HA_Close_Frozen', 'HA_Close', 'Low', 'High', 'Candle_Range', 
    'Market_Buy_Vol', 'Market_Sell_Vol', 'Total_Volume', 'Barrier_Level', 
    '15M_Delta_Momentum', 'Instant_Kinematic_Signal'
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1H HA Close',
    'HA_Close': '15M HA Close',
    'Low': 'Candle Low',
    'High': 'Candle High',
    'Candle_Range': 'Candle Range ($)',
    'Market_Buy_Vol': 'Real Buy Vol (BTC)',
    'Market_Sell_Vol': 'Real Sell Vol (BTC)',
    'Total_Volume': 'Total Vol (BTC)',
    'Barrier_Level': 'Active Barrier',
    '15M_Delta_Momentum': '15M Delta Momentum',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

# Safe Numeric Rounding Fix (Prevents TypeError on None/NaNs)
num_cols = [
    '1H HA Close', '15M HA Close', 'Candle Low', 'Candle High', 
    'Candle Range ($)', 'Real Buy Vol (BTC)', 'Real Sell Vol (BTC)', 
    'Total Vol (BTC)', 'Active Barrier', '15M Delta Momentum'
]

for col in num_cols:
    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0.0).round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
