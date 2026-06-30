import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time

# Streamlit Page Configuration
st.set_page_config(page_title="Hourly Nifty BeES Tracker", layout="wide")
st.title("🎯 Nifty BeES Hourly Institutional Tracker")
st.subheader("Every-Candle 1-Hour Matrix From 1st Jan 2025 to June 2026")

st.write("---")
st.write("### 8-Step Verification Progress:")

# STEP 1: Institutional Hourly Window Filter
def check_hourly_window(timestamp):
    h = timestamp.hour
    # Stock market hourly candles are split as: 9:15-10:15, 10:15-11:15, etc.
    # We focus on opening hours and afternoon European absorption blocks
    if h in [9, 10]:
        return "Morning_Momentum"
    elif h in [13, 14]:
        return "European_Absorption"
    return "No_Zone"

st.success("Step 1: Hourly Institutional Time Windows Locked.")

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

# STEP 3: Pure Intraday VWAP Calculation (Day-wise Reset Engine)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 3: Cumulative Hourly VWAP Layers Plotted.")

# STEP 4: Hourly Volume Benchmark (1.8x Multiplier over 20-period Average)
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 1.8)
st.success("Step 4: 1.8x Hourly Volume Breakthrough Filters Active.")

# STEP 5: Price Range Spread Filter (1.3x over 20-period Average)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=20, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.3)
st.success("Step 5: Price Action Volatility Filters Configured.")

# STEP 6 & 7: Filtration and Condition Matrix Alignment
df['Zone'] = [check_hourly_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
df['Is_Bullish'] = df['Close'] > df['Open']

# Core Rules: Price > VWAP + Bullish Hourly Candle + Volume Spike + Correct Time Window
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP']) & df['Is_Bullish']
st.success("Step 6 & 7: Noise Filter Applied. Final Matrix Compiled.")

# STEP 8: Final Count & Interactive Render
st.write("---")
st.header("📋 8-STEP HOURLY CANDLE-WISE MATRIX REPORT")

display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Institutional Block'] = df['Zone']
display_df['Close Price'] = df['Close'].round(2)
display_df['VWAP Level'] = df['VWAP'].round(2)
display_df['Hourly Volume'] = df['Volume'].astype(int)
display_df['Avg Vol'] = df['Vol_MA'].astype(int)
display_df['Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['Signal']]

total_signals = int(df['Signal'].sum())
st.metric(label="Total Long-Term Institutional Breaks Verified", value=total_signals)

st.write("### Every 1-Hour Candle Breakdown Table (Scroll to scan):")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Filtered Institutional Entry Summary (1-Hour Breakouts):")
signals_only = display_df[display_df['Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Final Verification Complete. Long-term hourly matrix mapped successfully!")
else:
    st.warning("1 Jan 2025 se lekar ab tak in strict parameters par koi clear alert match nahi hua. Aap parameters ko thoda relax kar sakte hain agar signals badhane hon.")
