import streamlit as st
import numpy as np
import pandas as pd
import requests

# Page Setup
st.set_page_config(page_title="BTC Kinematic Engine + Bid-Ask Candles", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Dual Engine + Historical Candle Bid/Ask Data")
st.caption("CoinGecko Public API Pipeline (Zero ISP/VPN Restrictions)")

# =====================================================================
# 🌐 COINGECKO / PUBLIC API ENGINE (FREQ FORMAT FIXED)
# =====================================================================
def fetch_btc_data_coingecko():
    """
    Fetches BTC Historical Candles via CoinGecko & Derives Order Flow Bid/Ask Estimates.
    Fixed Pandas frequency strings ('15min' & '1h').
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Fetch OHLC Data
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=30"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        
        if not isinstance(data, list):
            return pd.DataFrame(), pd.DataFrame()
            
        df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
        df['Open Time'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Kolkata')
        df.set_index('Open Time', inplace=True)
        df.drop(columns=['Timestamp'], inplace=True)
        
        # Synthetic Order Flow Estimates for Bid/Ask Spread & Volume
        df['Candle_Bid_Price'] = df['Low']
        df['Candle_Ask_Price'] = df['High']
        df['Candle_Spread'] = df['High'] - df['Low']
        
        # Simulated Volume Split
        range_diff = (df['High'] - df['Low']) / df['Close']
        df['Ask_Volume_Buy'] = np.round(100 * (1 + range_diff), 2)
        df['Bid_Volume_Sell'] = np.round(100 * (1 - range_diff), 2)
        
        # 1H & 15M Resampled Grid (Updated to Pandas-compatible frequency aliases)
        df_1h = df.resample('1h').interpolate(method='linear')
        df_15m = df.resample('15min').interpolate(method='linear')
        
        return df_1h, df_15m
    except Exception as e:
        st.error(f"Error fetching CoinGecko data: {e}")
        return pd.DataFrame(), pd.DataFrame()

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
with st.spinner("Fetching Market Data & Computing Indicators..."):
    df_1h_raw, df_15m_raw = fetch_btc_data_coingecko()

if df_1h_raw.empty or df_15m_raw.empty:
    st.error("🚨 Failed to fetch market data. Please check network connection.")
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
m2.metric("Candle Min Bid", f"${latest['Candle_Bid_Price']:,.2f}")
m3.metric("Candle Max Ask", f"${latest['Candle_Ask_Price']:,.2f}")
m4.metric("Candle Range", f"${latest['Candle_Spread']:.2f}")
m5.metric("Buy Vol (Est)", f"{latest['Ask_Volume_Buy']:.2f}")
m6.metric("Sell Vol (Est)", f"{latest['Bid_Volume_Sell']:.2f}")

st.markdown("---")

st.subheader("📋 Historical Candle Data (Includes Bid, Ask, Spread & Volume Split)")

# Format Columns for Clean Table
clean_cols = [
    '1H_HA_Close_Frozen', 'HA_Close', 'Candle_Bid_Price', 'Candle_Ask_Price', 'Candle_Spread', 
    'Ask_Volume_Buy', 'Bid_Volume_Sell', 'Barrier_Level', 
    '15M_Delta_Momentum', 'Instant_Kinematic_Signal'
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1H HA Close',
    'HA_Close': '15M HA Close',
    'Candle_Bid_Price': 'Candle Min Bid',
    'Candle_Ask_Price': 'Candle Max Ask',
    'Candle_Spread': 'Candle Range/Spread',
    'Ask_Volume_Buy': 'Market Buy Vol',
    'Bid_Volume_Sell': 'Market Sell Vol',
    'Barrier_Level': 'Active Barrier',
    '15M_Delta_Momentum': '15M Delta Momentum',
    'Instant_Kinematic_Signal': 'Kinematic Signal'
}, inplace=True)

# Safe Rounding Fix: Convert None/NaN to 0.0 before rounding to prevent TypeError
num_cols = [
    '1H HA Close', '15M HA Close', 'Candle Min Bid', 'Candle Max Ask', 
    'Candle Range/Spread', 'Market Buy Vol', 'Market Sell Vol', 
    'Active Barrier', '15M Delta Momentum'
]

for col in num_cols:
    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0.0).round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
