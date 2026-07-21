import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC 15M HA-HAM Engine", layout="wide")

# ==========================================
# 1. COMPLETE 90-DAYS BINANCE DATA FETCHER
# ==========================================
@st.cache_data(ttl=600)
def fetch_complete_90_days_data(symbol='BTCUSDT', interval='15m', days=90):
    url = "https://api.binance.com/api/v3/klines"
    
    # Calculate timestamps
    now = datetime.utcnow()
    start_dt = now - timedelta(days=days)
    
    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(now.timestamp() * 1000)
    
    all_data = []
    limit = 1000  # Binance Max Limit per Request
    current_start = start_time
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate estimated total loops
    total_candles_needed = (days * 24 * 4)  # 90 days * 24 hrs * 4 candles/hr = ~8640
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': limit
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                break
                
            all_data.extend(data)
            
            # Progress bar update
            progress = min(len(all_data) / total_candles_needed, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Fetching 90 Days Data... Loaded {len(all_data)} / ~{total_candles_needed} candles")
            
            # Move startTime to the last received timestamp + 1ms
            current_start = data[-1][0] + 1
            time.sleep(0.03)  # Small protection pause for API limits
            
        except Exception as e:
            st.error(f"Error fetching data batch: {e}")
            break

    progress_bar.empty()
    status_text.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# ==========================================
# 2. HEIKIN-ASHI & KALMAN FILTER ENGINE
# ==========================================
def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ha_open = [ (df['open'].iloc[0] + df['close'].iloc[0]) / 2 ]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)
        
    ha_df['HA_Close'] = ha_close
    ha_df['HA_Open'] = ha_open
    
    ha_df['HA_High'] = df[['high']].join(ha_df[['HA_Open', 'HA_Close']]).max(axis=1)
    ha_df['HA_Low'] = df[['low']].join(ha_df[['HA_Open', 'HA_Close']]).min(axis=1)
    return ha_df

def apply_kalman_filter(series, q=0.01, r=0.1):
    xhat = np.zeros(len(series))
    P = np.zeros(len(series))
    xhatminus = np.zeros(len(series))
    Pminus = np.zeros(len(series))
    K = np.zeros(len(series))
    
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    
    for k in range(1, len(series)):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + q
        K[k] = Pminus[k] / (Pminus[k] + r)
        xhat[k] = xhatminus[k] + K[k] * (series.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return pd.Series(xhat, index=series.index)

# ==========================================
# 3. LEVEL-FLIP SIGNAL GENERATOR
# ==========================================
def generate_signals_with_level_flip(df):
    df = calculate_heikin_ashi(df)
    
    # 15M HAM Calculation
    df['Kalman_HA_Close'] = apply_kalman_filter(df['HA_Close'])
    df['HAM'] = df['HA_Close'] - df['Kalman_HA_Close']
    
    # 1H Resampled HAM for Macro
    df_1h = df.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    df_1h = calculate_heikin_ashi(df_1h)
    df_1h['Kalman_HA_Close_1H'] = apply_kalman_filter(df_1h['HA_Close'])
    df_1h['HAM_1H'] = df_1h['HA_Close'] - df_1h['Kalman_HA_Close_1H']
    
    # Merge 1H Macro HAM
    df = pd.merge_asof(
        df.sort_values('timestamp'), 
        df_1h[['timestamp', 'HAM_1H']].rename(columns={'HAM_1H': 'h1_ham'}).sort_values('timestamp'), 
        on='timestamp', 
        direction='backward'
    )

    signals = ['NEUTRAL'] * len(df)
    barrier_levels = [None] * len(df)
    
    last_level = None          # Stores Resistance/Support Level
    active_state = None        # 'TOP' or 'BOTTOM'

    for i in range(1, len(df)):
        ha_close = df['HA_Close'].iloc[i]
        ha_open = df['HA_Open'].iloc[i]
        ha_high = df['HA_High'].iloc[i]
        ha_low = df['HA_Low'].iloc[i]
        
        m15_ham = df['HAM'].iloc[i]
        h1_curr = df['h1_ham'].iloc[i]
        h1_prev = df['h1_ham'].iloc[i-1] if i > 0 else h1_curr
        
        is_ha_red = ha_close < ha_open
        is_ha_green = ha_close >= ha_open

        # ----------------------------------------------------
        # 1. LEVEL BREAKOUT / INSTANT FLIP CHECK
        # ----------------------------------------------------
        if active_state == 'TOP' and last_level is not None:
            if ha_close > last_level:
                active_state = 'BOTTOM'
                last_level = ha_low   # Lock new support level
                signals[i] = '🟢 REAL BOTTOM'
                barrier_levels[i] = last_level
                continue

        elif active_state == 'BOTTOM' and last_level is not None:
            if ha_close < last_level:
                active_state = 'TOP'
                last_level = ha_high  # Lock new resistance level
                signals[i] = '🔴 REAL TOP'
                barrier_levels[i] = last_level
                continue

        # ----------------------------------------------------
        # 2. BASE HAM ENGINE CONDITIONS
        # ----------------------------------------------------
        if h1_curr > 0 and h1_curr < h1_prev and (m15_ham < 0 or is_ha_red):
            signals[i] = '🔴 REAL TOP'
            active_state = 'TOP'
            last_level = ha_high

        elif h1_curr < 0 and h1_curr > h1_prev and (m15_ham > 0 and is_ha_green):
            signals[i] = '🟢 REAL BOTTOM'
            active_state = 'BOTTOM'
            last_level = ha_low
            
        else:
            signals[i] = f"HOLD ({active_state})" if active_state else "HOLD"

        barrier_levels[i] = last_level

    df['Signal'] = signals
    df['Barrier_Level'] = barrier_levels
    return df

# ==========================================
# 4. STREAMLIT APP DISPLAY
# ==========================================
st.title("⚡ BTC 15M Real Top / Bottom Engine")
st.caption("3 Months Complete Data + Instant Level Flip Logic")

# Fetch Full 90 Days Data
df_raw = fetch_complete_90_days_data(symbol='BTCUSDT', interval='15m', days=90)

if df_raw.empty:
    st.error("Data load nahi ho paya, please refresh the page.")
else:
    df = generate_signals_with_level_flip(df_raw)

    st.success(f"Full Data Loaded! Total Candles: **{len(df):,}** | Range: **{df['timestamp'].min().strftime('%d %b %Y')}** to **{df['timestamp'].max().strftime('%d %b %Y')}**")

    # Metrics Row
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${latest['close']:,.2f}")
    col2.metric("Active Signal", latest['Signal'])
    col3.metric("Level Barrier", f"${latest['Barrier_Level']:,.2f}" if latest['Barrier_Level'] else "N/A")
    col4.metric("15M HAM", f"{latest['HAM']:.2f}")

    st.markdown("---")
    st.subheader("📋 Recent Signals & Level Flips")
    
    # Filter Only Signal Changes
    signals_df = df[df['Signal'].str.contains('REAL')].copy()
    display_cols = ['timestamp', 'close', 'HA_High', 'HA_Low', 'Signal', 'Barrier_Level']
    
    st.dataframe(
        signals_df[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
