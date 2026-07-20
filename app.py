import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import calendar
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="BTC Kinematic Peak-Trough ML Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Peak-Trough Reversal ML Engine")
st.caption("HAM Dynamic Peak/Trough Tracking + Post-Consolidation Confirmation Engine")

# =====================================================================
# MATHEMATICAL ENGINES
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

def get_month_expiry_date(dt_val):
    year = dt_val.year
    month = dt_val.month
    last_day = calendar.monthrange(year, month)[1]
    return f"{last_day} {dt_val.strftime('%b %Y')}"

# =====================================================================
# DATA INGESTION (2 YEARS 1H)
# =====================================================================
df = None
with st.spinner("Fetching BTC Data & Analyzing Peak/Trough Cycles..."):
    try:
        df = yf.download(tickers="BTC-USD", period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) > 300: 
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            df = df.iloc[:-1]
            
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

df = apply_heikin_ashi(df)
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

# Base Kinematics
normal_close = df_predict['Close'].to_numpy().flatten()
df_predict['Hurst_Normal'] = calculate_rolling_hurst_leak_free(normal_close, window=30)
kalman_base_normal = apply_kalman_filter_custom(normal_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal = apply_kalman_filter_custom(normal_close - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1)

df_predict['Kalman_Price'] = kalman_base_normal
df_predict['Weighted_Momentum'] = momentum_normal
df_predict['HAM_Normal'] = np.array(momentum_normal) * (df_predict['Hurst_Normal'].to_numpy() * 2.0)

ha_close = df_predict['HA_Close'].to_numpy().flatten()
df_predict['Hurst_HA'] = calculate_rolling_hurst_leak_free(ha_close, window=30)
kalman_base_ha = apply_kalman_filter_custom(ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha = apply_kalman_filter_custom(ha_close - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1)
df_predict['HAM_HeikinAshi'] = np.array(momentum_ha) * (df_predict['Hurst_HA'].to_numpy() * 2.0)

# Velocity & Acceleration
df_predict['_Velocity'] = df_predict['Kalman_Price'].diff(1)
df_predict['_Acceleration'] = df_predict['_Velocity'].diff(1)

# Basic Kinematic State Assignment
def assign_basic_state(row):
    v = row['_Velocity']
    a = row['_Acceleration']
    h = row['Hurst_Normal']
    
    if v < 0 or row['Close'] < row['Kalman_Price']:
        if a > 0 and h < 0.50:
            return 'GREEN' # Fake drop / Dip Buy
        elif a < 0 and h >= 0.50:
            return 'RED'   # Real Drop
        else:
            return 'CONSOLIDATION'
    elif v > 0 or row['Close'] > row['Kalman_Price']:
        if a > 0 and h >= 0.50:
            return 'GREEN' # Real Rally
        elif a < 0 and h < 0.50:
            return 'RED'   # Fake Rally Top
        else:
            return 'CONSOLIDATION'
    else:
        return 'CONSOLIDATION'

df_predict['Basic_State'] = df_predict.apply(assign_basic_state, axis=1)

# Near ATM Option Price Calculation (Rounded to nearest 100)
df_predict['Near_ATM_Strike'] = (df_predict['Close'] / 100.0).round() * 100

# =====================================================================
# ⚙️ EXACT PEAK / TROUGH REVERSAL ENGINE
# =====================================================================
n = len(df_predict)
ham_vals = df_predict['HAM_Normal'].to_numpy()
states = df_predict['Basic_State'].to_numpy()
dates = df_predict.index
atms = df_predict['Near_ATM_Strike'].to_numpy()

signals = ['⚪ NEUTRAL'] * n
option_preds = ['⚪ WAIT'] * n

# Dynamic State Tracking Variables
peak_ham = 0.0
trough_ham = 0.0
is_dropping_from_peak = False
is_rising_from_trough = False
seen_consolidation = False

for i in range(1, n):
    curr_ham = ham_vals[i]
    prev_ham = ham_vals[i-1]
    curr_state = states[i]
    
    # -------------------------------------------------------------
    # PATH A: POSITIVE HAM SIDE (TRACKING PEAK -> DROP -> CONSOLIDATION -> RED)
    # -------------------------------------------------------------
    if curr_ham > 0:
        # Track New Maximum Peak
        if curr_ham > peak_ham:
            peak_ham = curr_ham
            is_dropping_from_peak = False
            seen_consolidation = False
        # Peak Reached & HAM started dropping (e.g., 600 -> 500 -> 400)
        elif curr_ham < prev_ham and peak_ham > 50:
            is_dropping_from_peak = True
            
        if is_dropping_from_peak:
            if curr_state == 'CONSOLIDATION':
                seen_consolidation = True
            elif seen_consolidation and curr_state == 'RED':
                # 🎯 CONFIRMED TOP REVERSAL!
                signals[i] = '🔴 TOP REVERSAL CONFIRMED'
                expiry_dt = get_month_expiry_date(dates[i])
                option_preds[i] = f"SELL {int(atms[i])} CE (Zero on {expiry_dt})"
                
                # Reset Tracker for Next Cycle
                peak_ham = 0.0
                is_dropping_from_peak = False
                seen_consolidation = False

    # -------------------------------------------------------------
    # PATH B: NEGATIVE HAM SIDE (TRACKING TROUGH -> RISE -> CONSOLIDATION -> GREEN)
    # -------------------------------------------------------------
    elif curr_ham < 0:
        # Track New Deepest Trough
        if curr_ham < trough_ham:
            trough_ham = curr_ham
            is_rising_from_trough = False
            seen_consolidation = False
        # Trough Reached & HAM started rising (e.g., -600 -> -500 -> -400)
        elif curr_ham > prev_ham and trough_ham < -50:
            is_rising_from_trough = True
            
        if is_rising_from_trough:
            if curr_state == 'CONSOLIDATION':
                seen_consolidation = True
            elif seen_consolidation and curr_state == 'GREEN':
                # 🎯 CONFIRMED BOTTOM REVERSAL!
                signals[i] = '🟢 BOTTOM REVERSAL CONFIRMED'
                expiry_dt = get_month_expiry_date(dates[i])
                option_preds[i] = f"SELL {int(atms[i])} PE (Zero on {expiry_dt})"
                
                # Reset Tracker for Next Cycle
                trough_ham = 0.0
                is_rising_from_trough = False
                seen_consolidation = False

df_predict['Instant_Kinematic_Signal'] = signals
df_predict['ML_ATM_Option_Expiry_Prediction'] = option_preds

# Clean NaN
df_predict.dropna(subset=['Hurst_Normal', 'HAM_Normal'], inplace=True)

latest = df_predict.iloc[-1]
latest_time = df_predict.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 VISUAL DISPLAY
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest['Instant_Kinematic_Signal']
    opt_pred = latest['ML_ATM_Option_Expiry_Prediction']
    
    if 'BOTTOM' in sig or 'GREEN' in sig:
        st.success(f"### Live Signal ({latest_time})\n# {sig}\n\n**Option Recommendation:** `{opt_pred}`")
    elif 'TOP' in sig or 'RED' in sig:
        st.error(f"### Live Signal ({latest_time})\n# {sig}\n\n**Option Recommendation:** `{opt_pred}`")
    else:
        st.warning(f"### Live Signal ({latest_time})\n# {sig}\n\n**Option Recommendation:** `{opt_pred}`")

with col_s2:
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Close", f"${latest['Close']:,.2f}")
    m2.metric("Weighted Momentum", f"{latest['Weighted_Momentum']:.2f}")
    m3.metric("Pure HAM Score", f"{latest['HAM_Normal']:.2f}")

st.markdown("---")

# Clean Table Display
st.subheader("📋 Peak-Trough Kinematic Matrix & Near-ATM Expiration Table")
clean_cols = ['Close', 'HA_Close', 'Kalman_Price', 'Weighted_Momentum', 'Hurst_Normal', 'HAM_Normal', 'HAM_HeikinAshi', 'Instant_Kinematic_Signal', 'ML_ATM_Option_Expiry_Prediction']
display_df = df_predict[clean_cols].copy()

for c in ['Close', 'HA_Close', 'Kalman_Price', 'Weighted_Momentum', 'Hurst_Normal', 'HAM_Normal', 'HAM_HeikinAshi']:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
