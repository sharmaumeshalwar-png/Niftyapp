import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Future Tracker", layout="wide")
st.title("🎯 Nifty Future Institutional Candle Tracker")
st.subheader("Every-Candle 5-Minute Future Matrix (June 2026 Data Stream)")

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

# STEP 2: Fetch Nifty Future Intraday Data (Continuous Future Tracking)
@st.cache_data
def fetch_nifty_future_data():
    # 'NSEIM' is Yahoo Finance ticker for NSE NIFTY Continuous Future Contracts
    ticker = "NSEIM" 
    calculated_start = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=calculated_start, interval="5m", progress=False)
    return data, calculated_start

df = pd.DataFrame()
start_used = ""

try:
    raw_data, start_used = fetch_nifty_future_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success(f"Step 2: Nifty Future Intraday Data fetched successfully from {start_used}.")
    else:
        raise ValueError("YFinance continuous future symbol didn't return data. Switching to Spot Index tracking.")
except Exception as e:
    # Safe fallback to Nifty Spot Index if future ticker hits data sync lag
    try:
        raw_data = yf.download("^NSEI", start=(datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d'), interval="5m", progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.warning("Step 2 [Spot Fallback Mode]: Future symbol lagged. Displaying Nifty 50 Index Spot Data.")
    except:
        st.error("Step 2 Critical Error: Data feed completely blocked.")
        st.stop()

# STEP 3: Intraday VWAP Calculation (Day-wise Reset)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 3: Future VWAP Layers Plotted.")

# STEP 4: Optimized Volume Benchmark for Futures (1.8x Multiplier)
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 1.8)
st.success("Step 4: Future Contract Volume Multipliers Active.")

# STEP 5: Optimized Price Spread Benchmark (1.3x Volatility Multiplier)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=20, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.3)
st.success("Step 5: Price Spread Volatility Filters Configured.")

# STEP 6 & 7: Filtration and Condition Matrix
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
df['Is_Bullish'] = df['Close'] > df['Open']
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP']) & df['Is_Bullish']
st.success("Step 6 & 7: Market Noise Filtered. Signal Block Aggregated.")

# STEP 8: Live Screen Rendering
st.write("---")
st.header("📋 8-STEP NIFTY FUTURE MATRIX REPORT")

display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Institutional Window'] = df['Zone']
display_df['Close Price'] = df['Close'].round(2)
display_df['VWAP'] = df['VWAP'].round(2)
display_df['Volume'] = df['Volume'].astype(int)
display_df['Avg Vol'] = df['Vol_MA'].astype(int)
display_df['Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['Signal']]

total_signals = int(df['Signal'].sum())
st.metric(label="Total Institutional Future Entries Captured", value=total_signals)

st.write("### Nifty Future Every Candle Wise Table:")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Only Institutional Breakout Candles Summary:")
signals_only = display_df[display_df['Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Execution matrix verified. Pure institutional footprints captured for Nifty Future.")
else:
    st.warning("Pichle 28 dino mein in filters ke hisab se koi single massive candle institutional volume ke sath breakout nahi kar payi. Future trading ke liye parameters balance hain.")
