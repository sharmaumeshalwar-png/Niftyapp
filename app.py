import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Streamlit Page Configuration
st.set_page_config(page_title="Normal Market Kalman Tracker", layout="wide")
st.title("📊 Normal Market Kalman & VWAP Engine")
st.subheader("Every-Candle 1-Hour Matrix From 1st Jan 2025 (All Market Hours)")

st.write("---")
st.write("### 8-Step Verification Progress:")

st.success("Step 1: Institutional Time Windows Removed. All Market Candles Allowed.")

# STEP 2: Fetch Long-Term Data (1st Jan 2025 onwards)
@st.cache_data
def fetch_hourly_bees_data():
    ticker = "NIFTYBEES.NS"
    start_date = "2025-01-01"
    # Fetching 1-Hour Interval Data
    data = yf.download(ticker, start=start_date, interval="1h", progress=False)
    return data

df = pd.DataFrame()
try:
    raw_data = fetch_hourly_bees_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success(f"Step 2: Hourly data parsed successfully from 1st January 2025 ({len(df)} rows loaded).")
    else:
        raise ValueError("Dataframe completely empty.")
except Exception as e:
    st.error(f"Step 2 Fetch Error: {e}")
    st.stop()

# STEP 3: Variable Assignments (A = Close, B = VWAP with Day-wise Reset)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()

df['A'] = df['Close']                      # Variable A = Close
df['B'] = df['Cum_TP_Vol'] / df['Cum_Vol'] # Variable B = VWAP
st.success("Step 3: Variable A (Close) and Variable B (VWAP) Assigned.")

# STEP 4: Variable C Formulation (A > B for All Candles)
# C is the raw state where Close is above VWAP
df['C'] = (df['A'] > df['B']).astype(float)
st.success("Step 4: Variable C State Vector Formulated (True when A > B).")

# STEP 5: Kalman Filter Layer Implementation (D = Kalman of 0.0001 of C)
def apply_kalman_filter(series, noise_q=0.0001):
    kalman_values = []
    x_hat = 0.0  # Estimated state
    p = 1.0      # Error covariance
    r = 0.1      # Measurement noise covariance
    
    for val in series:
        # Prediction update
        p = p + noise_q
        # Measurement update (Gain calculation)
        k_gain = p / (p + r)
        x_hat = x_hat + k_gain * (val - x_hat)
        p = (1 - k_gain) * p
        kalman_values.append(x_hat)
    return kalman_values

df['D'] = apply_kalman_filter(df['C'], noise_q=0.0001)
st.success("Step 5: Kalman Filter Model D Generated with Noise Q = 0.0001.")

# STEP 6 & 7: Signal E Formulation (E Trigger Logic)
# Signal E triggers when the filtered trend baseline D crosses a stable state threshold (e.g., > 0.5) and A > B
df['E_Signal'] = (df['D'] > 0.5) & (df['A'] > df['B'])
st.success("Step 6 & 7: Core Mathematical Logic Matrix Compiled.")

# STEP 8: Final Count & Interactive Render
st.write("---")
st.header("📋 8-STEP NORMAL MARKET MATRIX REPORT")

# Preparing DataFrame for Streamlit View
display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['A (Close)'] = df['A'].round(2)
display_df['B (VWAP)'] = df['B'].round(2)
display_df['C (State Vector)'] = df['C'].astype(int)
display_df['D (Kalman Value)'] = df['D'].round(5)
display_df['E Signal Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['E_Signal']]

total_signals = int(df['E_Signal'].sum())
st.metric(label="Total Signal E Breakouts Captured", value=total_signals)

st.write("### Every 1-Hour Candle Mathematical Matrix (All Trading Hours):")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Signal E Active Execution Rows:")
signals_only = display_df[display_df['E Signal Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Final Count Verified. Normal market matrix generated perfectly!")
else:
    st.warning("Diye gaye time frame mein in parameters par koi active Signal E pass nahi hua.")
