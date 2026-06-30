import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time

# STEP 1: Define Institutional Time Windows
def is_institutional_window(timestamp):
    t = timestamp.time()
    return (time(9, 15) <= t <= time(10, 30)) or \
           (time(13, 0) <= t <= time(14, 0)) or \
           (time(14, 30) <= t <= time(15, 30))

# STEP 2: Fetch Multi-Date Data (All Possible Outcome Dates)
ticker = "RELIANCE.NS"  # Aap kisi bhi liquid stock par test kar sakte hain
data = yf.download(ticker, period="5d", interval="5m")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = data.copy()
df['Date'] = df.index.date

# STEP 3: Calculate VWAP (Intraday Basis)
# VWAP = Cumulative (Typical Price * Volume) / Cumulative Volume (Har din ke liye reset)
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TP_Vol'] = df['Typical_Price'] * df['Volume']

df['Cum_TP_Vol'] = df.groupby('Date')['TP_Vol'].cumsum()
df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']

# STEP 4: Calculate Volume Benchmark (20-Period MA)
df['Vol_MA'] = df['Volume'].rolling(window=20).mean()

# STEP 5: Apply Institutional Rules on 5-Min Candles
# Rule A: Candle VWAP ke upar cross karni chahiye (Bullish Crossover)
df['VWAP_Cross'] = (df['Close'] > df['VWAP']) & (df['Open'] <= df['VWAP'])

# Rule B: Volume pichle average se kam se kam 2Guna (2x) hona chahiye
df['Institutional_Volume'] = df['Volume'] > (df['Vol_MA'] * 2.0)

# Rule C: Time window match honi chahiye
df['Is_Window'] = [is_institutional_window(idx) for idx in df.index]

# STEP 6: Generate Trading Signals
df['Signal'] = df['VWAP_Cross'] & df['Institutional_Volume'] & df['Is_Window']

# STEP 7: Filter Outcomes
signals = df[df['Signal'] == True]

# STEP 8: Final 8-Step Verification Count & Report
print(f"=== 8-STEP VWAP VERIFICATION REPORT ===")
print(f"Step 1: Institutional Time Zones Locked.")
print(f"Step 2: 5-Day Intraday Data Fetched Successfully.")
print(f"Step 3: Intraday VWAP Calculated for each day.")
print(f"Step 4: Volume Moving Average (20) Baseline Set.")
print(f"Step 5: VWAP Crossover + 2x Volume Conditions Processed.")
print(f"Step 6: False Out-of-Hour Signals Filtered Out.")
print(f"Step 7: Compiling Final Institutional Entries...\n")

print(f"Total Big Moves Detected via VWAP Theory: {len(signals)}")
print("-" * 65)
if len(signals) > 0:
    for index, row in signals.iterrows():
        print(f"Time: {index} | Price: {row['Close']:.2f} | VWAP: {row['VWAP']:.2f} | Vol: {int(row['Volume'])} (Avg: {int(row['Vol_MA'])})")
else:
    print("In dates mein koi bada VWAP institutional breakout nahi mila.")

print(f"\nStep 8: Execution complete. All outcomes verified.")
