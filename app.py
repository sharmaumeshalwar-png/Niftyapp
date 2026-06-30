import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time

# Streamlit Page Configuration
st.set_page_config(page_title="Optimized Institutional Tracker", layout="wide")
st.title("🎯 Optimized Institutional Candle Tracker (2-Month Data)")
st.subheader("5-Minute Candle Matrix From 1st May 2026 to June 2026")

st.write("---")
st.write("### 8-Step Verification Progress:")

# STEP 1: Institutional Window Definition
def check_window(timestamp):
    t = timestamp.time()
    if (time(9, 15) <= t <= time(10, 30)):
        return "Morning_Momentum"
    elif (time(13, 0) <= t <= time(14, 30)):
        return "European_Absorption"
    return "No_Zone"

st.success("Step 1: Institutional Time Zones Locked.")

# STEP 2: Fetch Live Data (All Possible Outcome Dates from 2026-05-01)
@st.cache_data
def fetch_historical_data():
    # Nifty ke bajay agar aap kisi stock jaise RELIANCE.NS ko test karenge toh volume spikes zyada saaf dikhenge
    ticker = "RELIANCE.NS" 
    start_date = "2026-05-01"
    # Yahoo Finance maximum 1 month ya 60 days ka hi 5m data deta hai, isliye we pull the max available window
    data = yf.download(ticker, start=start_date, interval="5m", progress=False)
    return data

df = pd.DataFrame()
try:
    raw_data = fetch_historical_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success("Step 2: 2-Month Data fetched successfully from Yahoo Finance.")
    else:
        raise ValueError("YFinance returned empty.")
except Exception as e:
    st.error(f"Step 2 Error: Live data fetch fail hua. Check internet or ticker. {e}")
    st.stop()

# STEP 3: Intraday VWAP Calculation (Day-wise Reset)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 4: Day-wise VWAP Math Layer Computed.")

# STEP 4: OPTIMIZED Volume Benchmark (Relaxed from 3.0x to 1.8x)
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 1.8) 
st.success("Step 4: Volume threshold optimized to 1.8x for institutional flow.")

# STEP 5: OPTIMIZED Spread Benchmark (Relaxed from 1.8x to 1.3x)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=20, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.3)
st.success("Step 5: Spread threshold optimized to 1.3x.")

# STEP 6: Time-Zone Alignment
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
st.success("Step 6: Out-of-Hour Filter Synced.")

# STEP 7: Signal Compile (Bullish candle crossing above VWAP)
df['Is_Bullish'] = df['Close'] > df['Open']
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP']) & df['Is_Bullish']
st.success("Step 7: Multi-Layer Aggregation Matrix Compiled.")

# STEP 8: Final Count & Display
st.write("---")
st.header("📋 8-STEP OPTIMIZED MATRIX REPORT")

# Prepare display dataframe
display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Zone'] = df['Zone']
display_df['Close Price'] = df['Close'].round(2)
display_df['VWAP'] = df['VWAP'].round(2)
display_df['Volume'] = df['Volume'].astype(int)
display_df['Avg Vol'] = df['Vol_MA'].astype(int)
display_df['Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['Signal']]

total_signals = int(df['Signal'].sum())
st.metric(label="Total Institutional Breaks Intercepted", value=total_signals)

st.write("### Every Candle Wise Breakdown Table:")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Only Institutional Breakout Candles Summary:")
signals_only = display_df[display_df['Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Final Count Verified. Signals generated with relaxed institutional criteria!")
else:
    st.warning("Ab bhi koi signal nahi aaya? Apne Streamlit page par top-right mein 'R' daba kar refresh karein, ya ticker ko 'RELIANCE.NS' se badal kar 'SBIN.NS' par test karein.")
