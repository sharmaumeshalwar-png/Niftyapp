import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time

# Streamlit Page Configuration
st.set_page_config(page_title="Kalman Institutional Tracker", layout="wide")
st.title("🎯 Kalman Filter & VWAP Institutional Engine")
st.subheader("1-Hour Candle Matrix Frame (From 1st Jan 2025 Onwards)")

st.write("---")
st.write("### 8-Step Verification Progress:")

# STEP 1: Institutional Window Definition (Hourly Blocks)
def check_hourly_window(timestamp):
    h = timestamp.hour
    if h in [9, 10]:
        return "Morning_Momentum"
    elif h in [13, 14]:
        return "European_Absorption"
    return "No_Zone"

st.success("Step 1: Institutional Time Zones Locked.")

# STEP 2: Fetch Long-Term Data (1st Jan 2025 onwards)
@st.cache_data
def fetch_hourly_bees_data():
    ticker = "NIFTYBEES.NS"
    start_date = "2025-01-01"
    data = yf.download(ticker, start=start_date, interval="1h", progress=False)
    return data

df = pd.DataFrame()
try:
    raw_data = fetch_hourly_bees_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success(f"Step 2: Hourly data parsed successfully from 1st January 2025.")
    else:
        raise ValueError("Dataframe completely empty.")
except Exception as e:
    st.error(f"Step 2 Fetch Error: {e}")
    st.stop()

# STEP 3: Variable Assignments (A = Close, B = VWAP)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()

df['A'] = df['Close']                      # Variable A = Close
df['B'] = df['Cum_TP_Vol'] / df['Cum_Vol'] # Variable B = VWAP
st.success("Step 3: Variables A (Close) and B (VWAP) Assigned.")

# STEP 4: Variable C Formulation (A > B + Standard Volatility Constraints)
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 1.5)
df['Zone'] = [check_hourly_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"

# C is the boolean matrix where Close > VWAP inside Institutional Windows
df['C'] = (df['A'] > df['B']) & df['Valid_Zone'] & df['Is_Heavy_Volume']
st.success("Step 4: Variable C Core State Vector Formulated.")

# STEP 5: Kalman Filter Layer Implementation (D = Kalman of 0.0001 of C)
# Mathematical Equation for Kalman Filter Tracking:
# $$x_{k|k} = x_{k|k-1} + K_k (z_k - x_{k|k-1})$$
def apply_kalman_filter(series, noise_q=0.0001):
    kalman_values = []
    # Initial state
    x_hat = 0.0  # Estimated true state of C
    p = 1.0      # Estimation error covariance
    r = 0.1      # Measurement noise covariance
    
    for val in series:
        # Prediction update
        p = p + noise_q
        # Measurement update (Gain calculation)
        k_gain = p / (p + r)
        x_hat = x_hat + k_gain * (float(val) - x_hat)
        p = (1 - k_gain) * p
        kalman_values.append(x_hat)
    return kalman_values

df['D'] = apply_kalman_filter(df['C'], noise_q=0.0001)
st.success("Step 5: Kalman Filter Math Model D Generated with Process Noise Q = 0.0001.")

# STEP 6 & 7: Signal E Formulation (E Trigger Logic)
# Signal E triggers when Kalman filter confidence spikes up significantly
df['E_Signal'] = (df['D'] > 0.3) & (df['A'] > df['B'])
st.success("Step 6 & 7: Matrix Target E Signal Compiled.")

# STEP 8: Render Every Candle Wise Display
st.write("---")
st.header("📋 8-STEP KALMAL-VWAP EQUATION REPORT")

display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['A (Close)'] = df['A'].round(2)
display_df['B (VWAP)'] = df['B'].round(2)
display_df['C (State)'] = df['C'].astype(int)
display_df['D (Kalman Vector)'] = df['D'].round(5)
display_df['E Signal Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['E_Signal']]

total_signals = int(df['E_Signal'].sum())
st.metric(label="Total Institutional Breaks Verified via Kalman State", value=total_signals)

st.write("### Every 1-Hour Candle Breakdown Table:")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Signal E Execution Table:")
signals_only = display_df[display_df['E Signal Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Final Count Verified. Kalman state logic mapped successfully!")
else:
    st.warning("1 Jan 2025 se lekar ab tak Kalman parameters par koi active trace pass nahi hua. Noise constraint change karein.")
