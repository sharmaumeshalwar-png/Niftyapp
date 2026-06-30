import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# Streamlit Page Setup
st.set_page_config(page_title="Institutional Tracker", layout="wide")
st.title("🎯 Institutional Big Trader Footprint Tracker")
st.subheader("5-Minute Candle Analysis (9:15-10:30 AM & 1:00-2:30 PM)")

st.write("---")
st.write("### 8-Step Verification Progress:")

# STEP 1: Time Zone Locking
def check_window(timestamp):
    t = timestamp.time()
    if (time(9, 15) <= t <= time(10, 30)):
        return "Morning_Momentum"
    elif (time(13, 0) <= t <= time(14, 30)):
        return "European_Absorption"
    return "No_Zone"

st.success("Step 1: Institutional Time Zones Locked.")

# STEP 2: Local Data Generation (0% Blank Screen Guarantee)
base_date = datetime(2026, 6, 30, 9, 15)
times = [base_date + timedelta(minutes=5*i) for i in range(75)]

np.random.seed(42)
df = pd.DataFrame(index=times)
df['Open'] = np.random.uniform(23800, 23820, size=75)
df['High'] = df['Open'] + np.random.uniform(2, 8, size=75)
df['Low'] = df['Open'] - np.random.uniform(2, 5, size=75)
df['Close'] = df['Open'] + np.random.uniform(-4, 6, size=75)
df['Volume'] = np.random.uniform(5000, 15000, size=75)

# Injecting Big Trader Footprints
m_idx = 6  # 09:45 AM (Morning Window)
df.iloc[m_idx, df.columns.get_loc('Volume')] = 95000
df.iloc[m_idx, df.columns.get_loc('High')] = df.iloc[m_idx]['Open'] + 45
df.iloc[m_idx, df.columns.get_loc('Close')] = df.iloc[m_idx]['Open'] + 40

e_idx = 48 # 01:15 PM (European Window)
df.iloc[e_idx, df.columns.get_loc('Volume')] = 120000
df.iloc[e_idx, df.columns.get_loc('High')] = df.iloc[e_idx]['Open'] + 55
df.iloc[e_idx, df.columns.get_loc('Close')] = df.iloc[e_idx]['Open'] + 50

df['Date'] = df.index.date
st.success("Step 2: Local Stock Data Generated.")

# STEP 3: VWAP Calculation
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
st.success("Step 3: VWAP Line Computed.")

# STEP 4 & 5: Volume & Spread MA Benchmarks
df['Vol_MA'] = df['Volume'].rolling(window=15, min_periods=1).mean()
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 3.0)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=15, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 2.0)
st.success("Step 4 & 5: Volume & Volatility Benchmarks Applied.")

# STEP 6 & 7: Signal Filtering
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP'])
signals = df[df['Signal'] == True]
st.success("Step 6 & 7: Out-of-Hour Noise Filtered and Signals Compiled.")

# STEP 8: Display Results on Streamlit Screen
st.write("---")
st.header("📋 8-STEP BIG TRADER REPORT OUTPUT")

total_signals = len(signals)
st.metric(label="Total Institutional Trades Intercepted", value=total_signals)

if total_signals > 0:
    for index, row in signals.iterrows():
        with st.expander(f"🎯 TRAP DETECTED AT {index.strftime('%H:%M')} ({row['Zone']})", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Close Price", f"₹{row['Close']:.2f}")
            col2.metric("VWAP Price", f"₹{row['VWAP']:.2f}")
            col3.metric("Volume Spike",
