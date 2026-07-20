import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC Kinematic Instant Physics Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC/USDT Instant Kinematic & Acceleration Engine")
st.caption("Bar-by-Bar Instant Verification System | Velocity, Acceleration, Jerk & Hurst Analysis")

# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_fast(data_array, initial_p=10.0, q_val=0.01, r_val=0.01):
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

def calculate_rolling_hurst_fast(price_series, window=30):
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

# =====================================================================
# SYSTEM DATA INGESTION (2 YEARS, 1-HOUR CANDLES)
# =====================================================================
with st.spinner("Fetching Live 2 Years BTC Data..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 300: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            df = df.iloc[:-1] # Drop running live candle
            
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

# Split 50:50
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

# =====================================================================
# ⚡ INSTANT KINEMATICS COMPUTATION
# =====================================================================
normal_close = df_predict['Close'].to_numpy().flatten()

# 1. Position/Filter Price
df_predict['Kalman_Price'] = apply_kalman_filter_fast(normal_close)

# 2. Velocity (v) = 1st Derivative of Price
df_predict['Velocity'] = df_predict['Kalman_Price'].diff(1)

# 3. Acceleration (a) = 2nd Derivative (Rate of change of velocity)
df_predict['Acceleration'] = df_predict['Velocity'].diff(1)

# 4. Jerk (j) = 3rd Derivative (Rate of change of acceleration)
df_predict['Jerk'] = df_predict['Acceleration'].diff(1)

# 5. Fast Hurst Exponent
df_predict['Hurst'] = calculate_rolling_hurst_fast(normal_close, window=30)

# =====================================================================
# ⚡ SINGLE CANDLE INSTANT VERIFICATION LOGIC (at 635 level)
# =====================================================================
def instant_kinematic_decision(row):
    v = row['Velocity']
    a = row['Acceleration']
    h = row['Hurst']
    
    # Negative Velocity (Price Dropping)
    if v < 0:
        # Condition 1: Acceleration is Positive/Slowing Down & Mean-Reverting Hurst -> TRAP DROP!
        if a > 0 and h < 0.50:
            return '🟢 FAKE DROP / TRAP PASS (BUY DIP)'
        
        # Condition 2: Acceleration is Negative (Speeding Up Downwards) & Persistent Hurst -> REAL CASCADE DROP!
        elif a < 0 and h >= 0.50:
            return '🔴 ACCELERATED REAL DROP (EXIT / SHORT)'
        
        # Condition 3: Decelerating fall but high persistence
        elif a > 0 and h >= 0.50:
            return '⚠️ ABSORPTION / CONSOLIDATING DIP'
            
        else:
            return '🟡 NEUTRAL DOWNWARD PRESSURE'
            
    # Positive Velocity (Price Rising)
    elif v > 0:
        if a > 0 and h >= 0.50:
            return '🟢 ACCELERATED REAL RALLY'
        elif a < 0 and h < 0.50:
            return '🔴 FAKE RALLY / EXHAUSTION TOP'
        else:
            return '🟢 BULLISH DYNAMICS'
            
    else:
        return '⚪ SIDEWAYS / BALANCED'

df_predict['Kinematic_Signal'] = df_predict.apply(instant_kinematic_decision, axis=1)

df_predict.dropna(subset=['Velocity', 'Acceleration', 'Jerk', 'Hurst'], inplace=True)

latest = df_predict.iloc[-1]
latest_time = df_predict.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 UI DISPLAY
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Kinematic_Signal']
    if 'BUY' in sig or 'RALLY' in sig:
        st.success(f"### Instant Signal ({latest_time})\n# {sig}")
    elif 'REAL DROP' in sig or 'FAKE RALLY' in sig:
        st.error(f"### Instant Signal ({latest_time})\n# {sig}")
    else:
        st.warning(f"### Instant Signal ({latest_time})\n# {sig}")

with col_s2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Close", f"${latest['Close']:,.2f}")
    m2.metric("Velocity (v)", f"{latest['Velocity']:.2f}")
    m3.metric("Acceleration (a)", f"{latest['Acceleration']:.2f}")
    m4.metric("Hurst (H)", f"{latest['Hurst']:.2f}")

st.markdown("---")

# Data Table
st.subheader("📋 Bar-by-Bar Instant Kinematics Matrix (IST)")
clean_cols = ['Close', 'Kalman_Price', 'Velocity', 'Acceleration', 'Jerk', 'Hurst', 'Kinematic_Signal']
display_df = df_predict[clean_cols].copy()

for c in ['Close', 'Kalman_Price', 'Velocity', 'Acceleration', 'Jerk', 'Hurst']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
