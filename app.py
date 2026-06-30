import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, time

# Streamlit Page Configuration
st.set_page_config(page_title="Historical Candle Tracker", layout="wide")
st.title("📊 Every-Candle Institutional Tracker (Historical Loop)")
st.subheader("5-Minute Candle Matrix From 1st May 2026 onwards")

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

# STEP 2: Fetch Multi-Date Data (All Possible Outcome Dates starting 2026-05-01)
@st.cache_data # Data baar-baar download na ho aur crash na kare, isliye caching lagayi hai
def fetch_historical_data():
    ticker = "^NSEI" # Nifty 50 Index (Aap RELIANCE.NS bhi likh sakte hain)
    start_date = "2026-05-01"
    # Fetching 5-minute data
    data = yf.download(ticker, start=start_date, interval="5m", progress=False)
    return data

try:
    raw_data = fetch_historical_data()
    if raw_data.empty:
        st.error("Data empty mila. Market holidays check karein.")
        st.stop()
        
    # Multi-level columns fix (Error 87 permanent solution)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
        
    df = raw_data.copy()
    st.success("Step 2: Historical Data Fetched successfully from 1st May 2026.")
except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()

# STEP 3: Intraday VWAP Calculation (Day-wise Reset Loop)
df['Date'] = df.index.date
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 3: Day-wise Resetted VWAP Line Computed.")

# STEP 4: Volume Moving Average (15-period rolling)
df['Vol_MA'] = df['Volume'].rolling(window=15, min_periods=1).mean()
st.success("Step 4: Moving Volume Benchmark Activated.")

# STEP 5: Apply Institutional Rules
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 3.0) # 3x Volume rule
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=15, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.8) # Volatility spread
st.success("Step 5: 3x Vol + 1.8x Spread Formulas Synced.")

# STEP 6: Time-Zone Alignment
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
st.success("Step 6: Off-Market Noise Filtered.")

# STEP 7: Signal Compile
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP'])
st.success("Step 7: Multi-Layer Matrix Ready.")

# STEP 8: Render Every Candle Wise Display
st.write("---")
st.header("📋 8-STEP HISTORICAL CANDLE-WISE MATRIX")

# Re-formatting dataframe for clean presentation
display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Zone'] = df['Zone']
display_df['Close Price'] = df['Close'].round(2)
display_df['VWAP'] = df['VWAP'].round(2)
display_df['Volume'] = df['Volume'].astype(int)
display_df['Avg Vol'] = df['Vol_MA'].astype(int)
display_df['Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['Signal']]

# Filter signals count
total_signals = int(df['Signal'].sum())
st.metric(label="Total Institutional Breaks Verified (Since May 1)", value=total_signals)

# Every Candle Grid
st.write("### Every Candle Breakdown (Scroll Down to View All rows):")
st.dataframe(display_df, use_container_width=True)

# Separate Summary for Instant Entry Check
st.write("---")
st.subheader("⚡ Only Institutional Breakout Candles Summary:")
signals_only = display_df[display_df['Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Final Count Verified. Multi-date simulation rendered
