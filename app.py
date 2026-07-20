import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC 2Y 1H Kinematic Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC/USDT 2-Year Kinematic Engine (1H Candles)")
st.caption("Pure Price Kinematics | 1-Hour Timeframe | 2-Year Horizon (50:50 Split) in IST")

# =====================================================================
# MATHEMATICAL ENGINES (Pure Price Kinematics)
# =====================================================================
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
    """30-Window for Fast Bar-by-Bar Kinematic Response"""
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

def apply_heikin_ashi(df_in):
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
# SYSTEM DATA INGESTION (2 YEARS, 1-HOUR CANDLES)
# =====================================================================
df = None
with st.spinner("Fetching 2 Years 1-Hour Bitcoin Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 300: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            df = df.iloc[:-1] # Live running candle safety
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            st.error("🚨 Error: Insufficient Data")
            st.stop()
    except Exception as e:
        st.error(f"🚨 API Failure: {e}")
        st.stop()

# =====================================================================
# ⚡ PURE KINEMATICS ENGINE
# =====================================================================
df = apply_heikin_ashi(df)

# Exact 50:50 Train & Predict Split
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close = df_predict['Close'].to_numpy().flatten()
df_predict['Hurst_Normal'] = calculate_rolling_hurst_leak_free(normal_close, window=30)
kalman_base_normal = apply_kalman_filter_custom(normal_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal = apply_kalman_filter_custom(normal_close - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Kalman_Price'] = kalman_base_normal
df_predict['Weighted_Momentum'] = momentum_normal

# PURE HAM FORMULA: Momentum * (Hurst * 2.0)
df_predict['HAM_Normal'] = np.array(momentum_normal) * (df_predict['Hurst_Normal'].to_numpy() * 2.0)

# --- PATH B: HEIKIN-ASHI CANDLE KINEMATICS ---
ha_close = df_predict['HA_Close'].to_numpy().flatten()
df_predict['Hurst_HA'] = calculate_rolling_hurst_leak_free(ha_close, window=30)
kalman_base_ha = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha = apply_kalman_filter_custom(ha_close - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['HAM_HeikinAshi'] = np.array(momentum_ha) * (df_predict['Hurst_HA'].to_numpy() * 2.0)

# =====================================================================
# ⚙️ BACKEND KINEMATIC DYNAMICS (ACCELERATION & TRAP ENGINE)
# =====================================================================
# Hidden Backend Computations for Velocity & Acceleration
df_predict['_Velocity'] = df_predict['Kalman_Price'].diff(1)
df_predict['_Acceleration'] = df_predict['_Velocity'].diff(1)

def instant_kinematic_decision(row):
    v = row['_Velocity']
    a = row['_Acceleration']
    h = row['Hurst_Normal']
    mom = row['Weighted_Momentum']
    
    # Negative Velocity (Price Dropping)
    if v < 0 or row['Close'] < row['Kalman_Price']:
        if a > 0 and h < 0.50:
            return '🟢 FAKE DROP / TRAP PASS (BUY DIP)'
        elif a < 0 and h >= 0.50:
            return '🔴 ACCELERATED REAL DROP (EXIT / SHORT)'
        elif a > 0 and h >= 0.50:
            return '⚠️ ABSORPTION / CONSOLIDATING DIP'
        else:
            return '🔴 SELL'
            
    # Positive Velocity (Price Rising)
    elif v > 0 or row['Close'] > row['Kalman_Price']:
        if a > 0 and h >= 0.50:
            return '🟢 ACCELERATED REAL RALLY'
        elif a < 0 and h < 0.50:
            return '🔴 FAKE RALLY / EXHAUSTION TOP'
        else:
            return '🟢 BUY'
            
    else:
        return '⚪ NEUTRAL / PAUSE'

df_predict['Instant_Kinematic_Signal'] = df_predict.apply(instant_kinematic_decision, axis=1)

# Clean NaN
df_predict.dropna(subset=['Hurst_Normal', 'Hurst_HA'], inplace=True)

latest = df_predict.iloc[-1]
latest_time = df_predict.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 VISUAL DISPLAY (NO CHART, ONLY TABLE)
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Instant_Kinematic_Signal']
    if 'BUY' in sig or 'RALLY' in sig:
        st.success(f"### 1H Kinematic Signal ({latest_time})\n# {sig}")
    elif 'DROP' in sig or 'EXHAUSTION' in sig or 'SELL' in sig:
        st.error(f"### 1H Kinematic Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### 1H Kinematic Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Close", f"${latest['Close']:,.2f}")
    m2.metric("Weighted Momentum", f"{latest['Weighted_Momentum']:.2f}")
    m3.metric("Pure HAM Score", f"{latest['HAM_Normal']:.2f}")

st.markdown("---")

# Original Clean Table Structure Preserved
st.subheader("📋 Pure Kinematic Matrix (1-Hour Candles, IST)")
clean_cols = ['Close', 'HA_Close', 'Kalman_Price', 'Weighted_Momentum', 'Hurst_Normal', 'HAM_Normal', 'HAM_HeikinAshi', 'Instant_Kinematic_Signal']
display_df = df_predict[clean_cols].copy()

for c in ['Close', 'HA_Close', 'Kalman_Price', 'Weighted_Momentum', 'Hurst_Normal', 'HAM_Normal', 'HAM_HeikinAshi']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
