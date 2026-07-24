import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write("🎯 **Pure Direct Crypto Signals:** Dual H.A.M. Matrix (Normal vs Heikin-Ashi) in IST")

# =====================================================================
# MATHEMATICAL ENGINES (Strictly Backward-Looking / No Leakage)
# =====================================================================
def apply_kalman_filter_custom(data_array: np.ndarray, initial_p=50.0, q_val=0.001, r_val=0.1) -> list:
    if len(data_array) == 0: 
        return []
    x, p = data_array[0], initial_p  
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1.0 - k) * p
        filtered_values.append(x)
    return filtered_values

def calculate_rolling_hurst_leak_free(price_series: np.ndarray, window=100) -> np.ndarray:
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

def apply_heikin_ashi(df_in: pd.DataFrame) -> pd.DataFrame:
    op = df_in['Open'].to_numpy().flatten()
    hi = df_in['High'].to_numpy().flatten()
    lo = df_in['Low'].to_numpy().flatten()
    cl = df_in['Close'].to_numpy().flatten()
    
    ha_close = (op + hi + lo + cl) / 4.0
    
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0
    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))
    
    df_out = df_in.copy()
    df_out['HA_Open'] = ha_open
    df_out['HA_High'] = ha_high
    df_out['HA_Low'] = ha_low
    df_out['HA_Close'] = ha_close
    return df_out

# =====================================================================
# 🛡️ DYNAMIC ROLLING INGESTION (Continuous 729-Day Rolling Window)
# =====================================================================
@st.cache_data(ttl=300)  # Auto-refreshes every 5 minutes to catch newly closed candles
def load_btc_data():
    # Dynamic UTC Start & End boundaries to prevent static API caching issues
    end_time = pd.Timestamp.now(tz='UTC')
    start_time = end_time - pd.Timedelta(days=729)  # Max ~730 days allowed by Yahoo for 1h interval
    
    raw_df = yf.download(
        tickers="BTC-USD", 
        start=start_time, 
        end=end_time, 
        interval="1h",
        progress=False
    )
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
        
    if len(raw_df) > 120: 
        raw_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        
        # Drop active running unclosed candle for zero leakage
        raw_df = raw_df.iloc[:-1] 
        
        # Convert to Indian Standard Time (IST)
        if raw_df.index.tz is None:
            raw_df.index = raw_df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            raw_df.index = raw_df.index.tz_convert('Asia/Kolkata')
            
        return raw_df
    return None

df = None
with st.spinner("Fetching Dynamic 2-Year Hourly Bitcoin Data from Yahoo Finance..."):
    try:
        df = load_btc_data()
        if df is None:
            st.error("🚨 Error: Insufficient data lines retrieved from Yahoo Finance.")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# ⚡ CORE TRANSFORMATIONS & DUAL KINEMATICS ENGINE
# =====================================================================
# Apply Heikin-Ashi logic first
df = apply_heikin_ashi(df)

# Strict 50:50 split matrix execution (Auto-adjusts as the rolling window advances)
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(f"🟢 **Synced & Secured {len(df_predict)} IST Bitcoin Candles (Dynamic Dynamic Rolling Engine Active)!**")

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close = df_predict['Close'].to_numpy().flatten()
df_predict['Hurst_Normal'] = calculate_rolling_hurst_leak_free(normal_close, window=100)
kalman_base_normal = apply_kalman_filter_custom(normal_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal = apply_kalman_filter_custom(normal_close - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['HAM_Normal'] = np.array(momentum_normal) * (df_predict['Hurst_Normal'].to_numpy() * 2.0)

# --- PATH B: HEIKIN-ASHI CANDLE KINEMATICS ---
ha_close = df_predict['HA_Close'].to_numpy().flatten()
df_predict['Hurst_HA'] = calculate_rolling_hurst_leak_free(ha_close, window=100)
kalman_base_ha = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha = apply_kalman_filter_custom(ha_close - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['HAM_HeikinAshi'] = np.array(momentum_ha) * (df_predict['Hurst_HA'].to_numpy() * 2.0)

# Drop NaNs created by rolling window initial states
df_predict.dropna(subset=['Hurst_Normal', 'Hurst_HA'], inplace=True)

# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    'Close',            # Normal Raw Close
    'HA_Close',         # Heikin-Ashi Close
    'Hurst_Normal',     # Hurst for normal candles
    'Hurst_HA',         # Hurst for HA candles
    'HAM_Normal',       # H.A.M on Normal Candles
    'HAM_HeikinAshi'    # H.A.M on Heikin-Ashi Candles
]
display_df = df_predict[clean_cols].copy()

for c in clean_cols:
    display_df[c] = display_df[c].round(2)

# Sort descending for active trading view (Latest IST candles on top)
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.subheader("📋 50:50 Split Pure Kinematic Analysis Matrix (Normal vs Heikin-Ashi Comparison)")
st.dataframe(display_df, use_container_width=True, height=650)
