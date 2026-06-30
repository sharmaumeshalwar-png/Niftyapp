import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

print("=== STARTING TRADER ENGINE ===", flush=True)

# STEP 1: Time Zone Locking
def check_window(timestamp):
    t = timestamp.time()
    if (time(9, 15) <= t <= time(10, 30)):
        return "Morning_Momentum"
    elif (time(13, 0) <= t <= time(14, 30)):
        return "European_Absorption"
    return "No_Zone"

print("Step 1: Time Windows Set.", flush=True)

# STEP 2: Pure Local Data Generation (No Internet/yfinance Needed)
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
m_idx = 6  # 09:45 AM
df.iloc[m_idx, df.columns.get_loc('Volume')] = 95000
df.iloc[m_idx, df.columns.get_loc('High')] = df.iloc[m_idx]['Open'] + 45
df.iloc[m_idx, df.columns.get_loc('Close')] = df.iloc[m_idx]['Open'] + 40

e_idx = 48 # 01:15 PM
df.iloc[e_idx, df.columns.get_loc('Volume')] = 120000
df.iloc[e_idx, df.columns.get_loc('High')] = df.iloc[e_idx]['Open'] + 55
df.iloc[e_idx, df.columns.get_loc('Close')] = df.iloc[e_idx]['Open'] + 50

df['Date'] = df.index.date
print("Step 2: Local Stock Data Generated.", flush=True)

# STEP 3: VWAP Calculation
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']
df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
print("Step 3: VWAP Line Computed.", flush=True)

# STEP 4: Volume MA (15 Period)
df['Vol_MA'] = df['Volume'].rolling(window=15, min_periods=1).mean()
print("Step 4: Volume Benchmark Set.", flush=True)

# STEP 5: Institutional Rules
df['Is_Heavy_Volume'] = df['Volume'] > (df['Vol_MA'] * 3.0)
df['Candle_Spread'] = df['High'] - df['Low']
df['Spread_MA'] = df['Candle_Spread'].rolling(window=15, min_periods=1).mean()
df['Is_Big_Spread'] = df['Candle_Spread'] > (df['Spread_MA'] * 2.0)
print("Step 5: Mathematical Rules Applied.", flush=True)

# STEP 6: Zone Filtering
df['Zone'] = [check_window(idx) for idx in df.index]
df['Valid_Zone'] = df['Zone'] != "No_Zone"
print("Step 6: Out-of-Hour Windows Filtered.", flush=True)

# STEP 7: Signal Triggers
df['Signal'] = df['Is_Heavy_Volume'] & df['Is_Big_Spread'] & df['Valid_Zone'] & (df['Close'] > df['VWAP'])
signals = df[df['Signal'] == True]
print("Step 7: Signals Analyzed.", flush=True)

# STEP 8: Final Verified Output
print("\n==================================================", flush=True)
print("        8-STEP BIG TRADER REPORT OUTPUT           ", flush=True)
print("==================================================", flush=True)
print("Total Signals Found: " + str(len(signals)), flush=True)
print("--------------------------------------------------", flush=True)

for index, row in signals.iterrows():
    t_str = index.strftime('%H:%M')
    print("🎯 TRAP TIME: " + t_str + " (Zone: " + str(row['Zone']) + ")", flush=True)
    print("   Close Price: " + str(round(row['Close'], 2)) + " | VWAP: " + str(round(row['VWAP'], 2)), flush=True)
    print("   Volume Spike: " + str(int(row['Volume'])) + " (Avg: " + str(int(row['Vol_MA'])) + ")", flush=True)
    print("--------------------------------------------------", flush=True)

print("\nStep 8: Final Count Verified. Screen Output Complete!", flush=True)
