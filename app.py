import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. 90-DAYS DATA FETCHER (BINANCE)
# ==========================================
def fetch_90_days_data(symbol='BTC/USDT', timeframe='15m', days=90):
    print(f"Fetching {days} days of {timeframe} data from Binance...")
    exchange = ccxt.binance()
    
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    all_ohlcv = []
    
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not len(ohlcv):
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

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
# 3. MAIN SIGNAL ENGINE WITH LEVEL FLIP
# ==========================================
def generate_signals_with_level_flip(df):
    df = calculate_heikin_ashi(df)
    
    # HAM Indicator Calculation
    df['Kalman_HA_Close'] = apply_kalman_filter(df['HA_Close'])
    df['HAM'] = df['HA_Close'] - df['Kalman_HA_Close']
    
    # 1H Resampled HAM for Macro Confirmation
    df_1h = df.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    df_1h = calculate_heikin_ashi(df_1h)
    df_1h['Kalman_HA_Close_1H'] = apply_kalman_filter(df_1h['HA_Close'])
    df_1h['HAM_1H'] = df_1h['HA_Close'] - df_1h['Kalman_HA_Close_1H']
    
    # Merge 1H HAM back to 15M dataframe
    df = pd.merge_asof(df.sort_values('timestamp'), 
                       df_1h[['timestamp', 'HAM_1H']].rename(columns={'HAM_1H': 'h1_ham'}).sort_values('timestamp'), 
                       on='timestamp', direction='backward')

    # Signal Array and Variables
    signals = ['NEUTRAL'] * len(df)
    last_level = None          # High/Low Invalidation Level
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
        # STEP A: LEVEL BREAKOUT / FLIP CHECK (HIGHEST PRIORITY)
        # ----------------------------------------------------
        if active_state == 'TOP' and last_level is not None:
            # AGAR 15M CLOSE TOP LEVEL KE UPAR GAYA -> INSTANT REAL BOTTOM
            if ha_close > last_level:
                active_state = 'BOTTOM'
                last_level = ha_low   # New Support Level
                signals[i] = '🟢 REAL BOTTOM'
                continue

        elif active_state == 'BOTTOM' and last_level is not None:
            # AGAR 15M CLOSE BOTTOM LEVEL KE NICHE GAYA -> INSTANT REAL TOP
            if ha_close < last_level:
                active_state = 'TOP'
                last_level = ha_high  # New Resistance Level
                signals[i] = '🔴 REAL TOP'
                continue

        # ----------------------------------------------------
        # STEP B: NORMAL ENGINE BASE SIGNALS
        # ----------------------------------------------------
        # Real Top Condition
        if h1_curr > 0 and h1_curr < h1_prev and (m15_ham < 0 or is_ha_red):
            signals[i] = '🔴 REAL TOP'
            active_state = 'TOP'
            last_level = ha_high   # Save Resistance Level

        # Real Bottom Condition
        elif h1_curr < 0 and h1_curr > h1_prev and (m15_ham > 0 and is_ha_green):
            signals[i] = '🟢 REAL BOTTOM'
            active_state = 'BOTTOM'
            last_level = ha_low    # Save Support Level
            
        else:
            signals[i] = 'HOLD / NO SIGNAL' if active_state is None else f"HOLD ({active_state})"

    df['Signal'] = signals
    df['Invalidation_Barrier'] = last_level
    return df

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == '__main__':
    # 1. Fetch 90 Days Data
    raw_df = fetch_90_days_data(symbol='BTC/USDT', timeframe='15m', days=90)
    
    # 2. Run Engine & Level Flip Logic
    final_df = generate_signals_with_level_flip(raw_df)
    
    # 3. Print Output Summary
    print("\n--- DATA RANGE ---")
    print(f"From: {final_df['timestamp'].min()}  To: {final_df['timestamp'].max()}")
    print(f"Total 15M Candles Processed: {len(final_df)}")
    
    print("\n--- RECENT 15 SIGNALS WITH LEVEL FLIP ---")
    cols_to_show = ['timestamp', 'open', 'high', 'low', 'close', 'Signal', 'Invalidation_Barrier']
    print(final_df[final_df['Signal'].str.contains('REAL')][cols_to_show].tail(15).to_string(index=False))
