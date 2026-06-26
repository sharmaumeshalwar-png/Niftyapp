import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. PARAMETERS & DATA FETCHING (2 YEARS LOCK)
# ==========================================
ticker = "^NSEI"  # Nifty 50 Index
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=730)  # 2 Years lookback

print(f"Fetching 1-Hour Nifty data from {start_date} to {end_date}...")

# ERROR FIX: group_by='ticker' हटाकर auto_adjust=True का उपयोग किया
df = yf.download(
    ticker, start=start_date, end=end_date, interval="1h", auto_adjust=True
)

if df.empty:
    raise ValueError("Data nahi mil paya. Ticker check karein.")

# ERROR FIX: Multi-index columns ko completely flatten/drop karne ka foolproof tareeka
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Data index properly clear karna
df = df.dropna()

# ==========================================
# 2. ADAPTIVE KALMAN FILTER IMPLEMENTATION
# ==========================================
def calculate_adaptive_kalman(data_df, q_base=0.01, r_base=1.0, velocity_period=3):
    # Arrays ko 1D/Flat matrix me convert karna explicit safely
    closes = data_df["Close"].to_numpy().flatten()
    highs = data_df["High"].to_numpy().flatten()
    lows = data_df["Low"].to_numpy().flatten()

    n = len(closes)
    kalman_output = np.zeros(n)

    # ATR Calculation
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])
        ),
    )
    atr = np.zeros(n)
    atr[1:] = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    atr[0] = atr[1] if atr[1] != 0 else 1.0
    atr = np.where(atr == 0, 1.0, atr)

    # Initial State
    x = closes[0]
    p = 1.0
    kalman_output[0] = x

    for i in range(1, n):
        # Calculate Price Velocity
        if i >= velocity_period:
            velocity = abs(closes[i] - closes[i - velocity_period])
        else:
            velocity = abs(closes[i] - closes[0])

        # ADAPTIVE LOGIC
        velocity_factor = velocity / atr[i]
        Q = q_base * (1.0 + velocity_factor * 5.0)
        R = r_base

        # Kalman Filter Formulas
        p = p + Q
        k_gain = p / (p + R)
        x = x + k_gain * (closes[i] - x)
        p = (1 - k_gain) * p

        kalman_output[i] = x

    # Assigning back to dataframe securely
    data_df = data_df.copy()
    data_df["Kalman"] = kalman_output
    data_df["ATR"] = atr
    return data_df


# Execute Adaptive Kalman
df = calculate_adaptive_kalman(df)

# ==========================================
# 3. STRATEGY LOGIC: c = a - b & SIGNALS
# ==========================================
df["a"] = df["Close"]
df["b"] = df["Kalman"]
df["c"] = df["a"] - df["b"]
df["c_scaled"] = df["c"] / df["ATR"]

df["Signal"] = 0
threshold = 1.5

# Signal Loop
c_vals = df["c"].to_numpy()
c_scaled_vals = df["c_scaled"].to_numpy()
signals = np.zeros(len(df))

for i in range(1, len(df)):
    if c_scaled_vals[i - 1] > threshold and c_vals[i] < c_vals[i - 1]:
        signals[i] = -1  # Sell Reversion
    elif c_scaled_vals[i - 1] < -threshold and c_vals[i] > c_vals[i - 1]:
        signals[i] = 1  # Buy Reversion

df["Signal"] = signals

# ==========================================
# 4. PRINT SUMMARY
# ==========================================
print("\n--- Strategy Analysis Complete ---")
print(f"Total Candles Analyzed: {len(df)}")
print(f"Total Short Signals: {len(df[df['Signal'] == -1])}")
print(f"Total Long Signals: {len(df[df['Signal'] == 1])}")
print("\nLast 5 Rows Data:")
print(df[["Close", "Kalman", "c", "c_scaled", "Signal"]].tail(5))
