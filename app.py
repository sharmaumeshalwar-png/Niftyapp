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
df = yf.download(ticker, start=start_date, end=end_date, interval="1h")

if df.empty:
    raise ValueError(
        "Data nahi mil paya. Internet connection ya ticker check karein."
    )

# Multi-index columns ko flatten karna (agar yfinance ka naya version ho)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()


# ==========================================
# 2. ADAPTIVE KALMAN FILTER IMPLEMENTATION
# ==========================================
def calculate_adaptive_kalman(df, q_base=0.01, r_base=1.0, velocity_period=3):
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    n = len(closes)
    kalman_output = np.zeros(n)

    # ATR Calculation for Normalization & Volatility
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])
        ),
    )
    atr = np.zeros(n)
    atr[1:] = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    atr[0] = atr[1] if atr[1] != 0 else 1.0
    atr = np.where(atr == 0, 1.0, atr)  # Zero division avoid karne ke liye

    # Initial State
    x = closes[0]  # Initial price estimate
    p = 1.0  # Initial error covariance

    kalman_output[0] = x

    for i in range(1, n):
        # Calculate Price Velocity
        if i >= velocity_period:
            velocity = abs(closes[i] - closes[i - velocity_period])
        else:
            velocity = abs(closes[i] - closes[0])

        # ADAPTIVE LOGIC: Velocity badhegi toh Q badhega
        # Isse filter fast ho jayega aur breakout me price ke sath chalega
        velocity_factor = velocity / atr[i]
        Q = q_base * (1.0 + velocity_factor * 5.0)  # 5.0 is a multiplier
        R = r_base

        # Kalman Filter Equations
        p = p + Q  # Predict error covariance
        k_gain = p / (p + R)  # Kalman Gain
        x = x + k_gain * (closes[i] - x)  # Update state estimate
        p = (1 - k_gain) * p  # Update error covariance

        kalman_output[i] = x

    df["Kalman"] = kalman_output
    df["ATR"] = atr
    return df


# Execute Adaptive Kalman
df = calculate_adaptive_kalman(df)

# ==========================================
# 3. STRATEGY LOGIC: c = a - b & SIGNALS
# ==========================================
df["a"] = df["Close"]  # Close Price
df["b"] = df["Kalman"]  # Kalman Filter
df["c"] = df["a"] - df["b"]  # Distance

# Normalized C score (Z-Score alternative using ATR)
df["c_scaled"] = df["c"] / df["ATR"]

# Signal Generation
# Entry tabhi hogi jab c_scaled strong bands ko cross karke hook back karega
df["Signal"] = 0  # 0 = No Trade, 1 = Buy (Reversion), -1 = Sell (Reversion)

threshold = 1.5  # Entry trigger threshold

for i in range(1, len(df)):
    # Short Trade (Mean Reversion): Price overextended upside but velocity cooling down
    if df["c_scaled"].iloc[i - 1] > threshold and df["c"].iloc[i] < df[
        "c"
    ].iloc[i - 1]:
        df.loc[df.index[i], "Signal"] = -1

    # Long Trade (Mean Reversion): Price overextended downside but hooking up
    elif df["c_scaled"].iloc[i - 1] < -threshold and df["c"].iloc[i] > df[
        "c"
    ].iloc[i - 1]:
        df.loc[df.index[i], "Signal"] = 1

# ==========================================
# 4. PRINT SUMMARY & SAMPLE SIGNALS
# ==========================================
print("\n--- Strategy Analysis Complete ---")
print(f"Total Candles Analyzed: {len(df)}")
print(f"Total Short Signals Generated: {len(df[df['Signal'] == -1])}")
print(f"Total Long Signals Generated: {len(df[df['Signal'] == 1])}")

# Display last 10 rows with signals
print("\nLast 10 Rows Data:")
print(df[["Close", "Kalman", "c", "c_scaled", "Signal"]].tail(10))
