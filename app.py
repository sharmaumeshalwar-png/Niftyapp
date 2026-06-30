import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Institutional Tracker", layout="wide")
st.title("🎯 Nifty Institutional Candle Tracker (No-Volume Fix)")
st.subheader("Every-Candle 5-Minute Matrix Using Price Action Spread")

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

# STEP 2: Fetch Nifty Spot Data (Highly Accurate Price Source)
@st.cache_data
def fetch_nifty_data():
    ticker = "^NSEI" # Nifty 50 Index Spot
    calculated_start = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=calculated_start, interval="5m", progress=False)
    return data, calculated_start

df = pd.DataFrame()
try:
    raw_data, start_used = fetch_nifty_data()
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        df = raw_data.copy()
        st.success(f"Step 2: Nifty Intraday Price Data fetched successfully.")
    else:
        raise ValueError("Data source returned empty.")
except Exception as e:
    st.error(f"Step 2 Error: {e}")
    st.stop()

# STEP 3: Substitute VWAP with 20-Period EMA (Institutional Baseline)
# Jab volume nahi hota, bade traders 20 EMA ko dynamic support maante hain
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
st.success("Step 3: Institutional 20-EMA Baseline Plotted (VWAP Alternative).")

# STEP 4 & 5: Price Spread as a Volume Substitute (The Volatility Spike Rule)
# High - Low badhna matlab large orders punch hue hain
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=20, min_periods=1).mean()

# Agar kisi 5-min candle ki size average se 1.8 गुना badi hai, toh wahi humara Volume Spike hai!
df['Is_Heavy_Activity'] = df['Candle_Spread'] > (df['Spread_MA'] * 1.8)
st.success("Step 4 & 5: Price Spread Volatility Analysed (1.8x Activity Filter).")

# STEP 6 & 7: Filtration and Signal Compilation
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
df['Is_Bullish'] = df['Close'] > df['Open']

# Rule: Big Price Spread + Above 20 EMA + Bullish + Right Institutional Time
df['Signal'] = df['Is_Heavy_Activity'] & df['Valid_Zone'] & (df['Close'] > df['EMA_20']) & df['Is_Bullish']
st.success("Step 6 & 7: Multi-Layer Aggregation Completed.")

# STEP 8: Live Screen Rendering
st.write("---")
st.header("📋 8-STEP NIFTY CANDLE-WISE MATRIX REPORT")

display_df = pd.DataFrame(index=df.index)
display_df['Date/Time'] = df.index.strftime('%Y-%m-%d %H:%M')
display_df['Institutional Window'] = df['Zone']
display_df['Close Price'] = df['Close'].round(2)
display_df['Baseline (20 EMA)'] = df['EMA_20'].round(2)
display_df['Candle Size (Points)'] = df['Candle_Spread'].round(2)
display_df['Avg Size (Points)'] = df['Spread_MA'].round(2)
display_df['Status'] = ["🎯 BUY ALERT" if s else "❌ No Action" for s in df['Signal']]

total_signals = int(df['Signal'].sum())
st.metric(label="Total Institutional Breaks Intercepted", value=total_signals)

st.write("### Nifty Every Candle Wise Table:")
st.dataframe(display_df, use_container_width=True)

st.write("---")
st.subheader("⚡ Only Institutional Breakout Candles Summary:")
signals_only = display_df[display_df['Status'] == "🎯 BUY ALERT"]

if not signals_only.empty:
    st.dataframe(signals_only, use_container_width=True)
    st.success("Step 8: Execution matrix verified. Nifty price action footprints captured successfully!")
else:
    st.warning("Pichle 28 dino mein in filters ke hisab se koi single massive candle institutional breakout nahi kar payi.")
