import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# STEP 1: Define Institutional Time Windows
def is_institutional_window(timestamp):
    t = timestamp.time()
    return (time(9, 15) <= t <= time(10, 30)) or \
           (time(13, 0) <= t <= time(14, 0)) or \
           (time(14, 30) <= t <= time(15, 30))

print("Step 1: Institutional Time Zones Locked.")

# STEP 2: Generate Safe/Fallback Data (All Possible Outcome Dates)
# Agar yfinance fail ho jaye, toh yeh blank screen ko bachaega
try:
    import yfinance as yf
    print("Step 2: Fetching Live Data via yfinance...")
    data = yf.download("RELIANCE.NS", period="2d", interval="5m", progress=False)
    if data.empty:
        raise ValueError("Data empty")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.copy()
except:
    print("Step 2 [Fallback]: Live data unavailable. Creating simulated institutional data...")
    base_time = datetime(2026, 6, 30, 9, 15)
    times = [base_time + timedelta(minutes=5*i) for i in range(75)]
    np.random.seed(42)
    
    # Simulated stock movement
    df = pd.DataFrame(index=times)
    df['Open'] = np.random.uniform(2500, 2520, size=75)
    df['High'] = df['Open'] + np.random.uniform(2, 10, size=75)
    df['Low'] = df['Open'] - np.random.uniform(2, 5, size=75)
    df['Close'] = df['Open'] + np.random.uniform(-5, 8, size=75)
    df['Volume'] = np.random.uniform(5000, 20000, size=75)
    
    # Injecting a fake institutional move at 1:15 PM (European Open)
    inst_idx = [i for i, t in enumerate(times) if t.hour == 13 and t.minute == 15][0]
    df.iloc[inst_idx, df.columns.get_loc('Close')] = df.iloc[inst_idx]['Open'] + 25 # Big green bar
    df.iloc[inst_idx, df.columns.get_loc('Volume')] = 150000 # Massive institutional volume

df['Date'] = df.index.date

# STEP 3: Calculate VWAP
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
print("Step 3: Intraday VWAP Calculated.")

# STEP 4: Volume Moving Average
df['Vol_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
print("Step 4: Volume Moving Average Baseline Set.")

# STEP 5: Apply Institutional Rules
df['VWAP_Cross'] = (df['Close'] > df['VWAP']) & (df['Open'] <= df['VWAP'])
df['Institutional_Volume'] = df['Volume'] > (df['Vol_MA'] * 2.0)
df['Is_Window'] = [is_institutional_window(idx) for idx in df.index]
print("Step 5: VWAP Crossover + 2x Volume Conditions Processed.")

# STEP 6: Filter Out-of-Hour Signals
df['Signal'] = df['VWAP_Cross'] & df['Institutional_Volume'] & df['Is_Window']
print("Step 6: False Out-of-Hour Signals Filtered Out.")

# STEP 7
