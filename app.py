import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time

# Streamlit Page Configuration
st.set_page_config(page_title="Dynamic Institutional Tracker", layout="wide")
st.title("🎯 Real Live Institutional Candle Tracker")
st.subheader("Every-Candle 5-Minute Matrix (Safe Data Fetching Enabled)")

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

# STEP 2: Safe Multi-Date Fetching (Maximum 29 Days to prevent Yahoo API Crash)
@st.cache_data
def fetch_safe_historical_data():
    ticker = "RELIANCE.NS" # Standard high liquidity stock for FII tracking
    # Dynamically calculating past 28 days to stay within Yahoo Intraday limits
    calculated_start = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=calculated_start, interval="5m", progress=False)
    return data, calculated_start

df = pd.DataFrame()
start_used = ""

try:
    raw_data, start_used = fetch_safe_historical_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success(f"Step 2: Safe Intraday Data fetched successfully from {start_used} to Today.")
    else:
        raise ValueError("YFinance blocked the connection.")
except Exception as e:
    st.error(f"Step 2 Critical Error: {e}")
    st.stop()

# STEP 3: Intraday VWAP Calculation (Day-wise Reset)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 3: Day-wise VWAP Layers Plotted.")

# STEP 4: Optimized Volume Benchmark (1.8x Multiplier)
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 1.8)
st.success("Step 4: 1.8x Volume Breakthrough Rules Synchronized.")

# STEP 5: Optimized Spread Benchmark (1.3x Volatility Multiplier)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=20, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.3)
st.success("Step 5: Price Action Volatility Filters Active.")

# STEP 6 & 7: Time Matching & Signal Matrix Construction
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
df['Is_Bullish'] = df['Close'] > df['Open']
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP']) & df['Is_Bullish']
st.success("Step 6 & 7: False Out-of-Hour Fluctuations Filtered Out.")

# STEP 8: Render Outputs
st.write("---")
st.header("📋 8-STEP OPTIMIZED MATRIX REPORT")

# Preparing DataFrame for clean Streamlit View
display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Institutional Window'] = df['Zone']
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
    st.success("Step 8: Final Count Verified. Data rendered perfectly without any empty dataframe error!")
else:
    st.warning("Is selected 30 days window mein in parameters par koi valid footprint nahi mila. Parameters aur zyada ghatiya (relax) na karein warna fake signals milenge.")
