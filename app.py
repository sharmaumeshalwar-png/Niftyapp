import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(page_title="BTC Master Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Pure Kinematic Action Master Engine")
st.write(
    "🎯 **Direct Binance Stream:** Dual H.A.M. Matrix (Normal vs Heikin-Ashi) in"
    " IST [2-Year Full Engine / Zero Leakage]"
)

# Sidebar Refresh Controls
st.sidebar.header("🔄 Binance Stream Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **Binance Direct Feed:** CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.001, r_val=0.1
):
  """Recursive Forward-Only Kalman Filter"""
  arr = np.asarray(data_array, dtype=float).flatten()
  if len(arr) == 0:
    return np.array([])
  x, p = arr[0], initial_p
  filtered_values = np.empty(len(arr))
  for i, z in enumerate(arr):
    p = p + q_val
    k = p / (p + r_val)
    x = x + k * (z - x)
    p = (1 - k) * p
    filtered_values[i] = x
  return filtered_values


def calculate_rolling_hurst_vectorized(price_series, window=100):
  """Backward-Looking Vectorized Rolling Hurst Exponent"""
  arr = np.asarray(price_series, dtype=float).flatten()
  s = pd.Series(arr)
  log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
  hurst_values = np.full(len(arr), 0.5)

  if len(log_returns) < window:
    return hurst_values

  windows = np.lib.stride_tricks.sliding_window_view(
      log_returns, window_shape=window
  )
  means = np.mean(windows, axis=1, keepdims=True)
  cum_dev = np.cumsum(windows - means, axis=1)

  r_val = np.ptp(cum_dev, axis=1)
  s_val = np.std(windows, axis=1) + 1e-10
  rs_ratio = r_val / s_val

  valid_mask = rs_ratio > 0
  h_calculated = np.full(len(rs_ratio), 0.5)
  h_calculated[valid_mask] = np.log(rs_ratio[valid_mask]) / np.log(window)

  hurst_values[window - 1 :] = np.clip(h_calculated, 0.0, 1.0)
  return hurst_values


def apply_heikin_ashi(df_in):
  """Vectorized Heikin-Ashi Transformation"""
  op = np.asarray(df_in["Open"], dtype=float).flatten()
  hi = np.asarray(df_in["High"], dtype=float).flatten()
  lo = np.asarray(df_in["Low"], dtype=float).flatten()
  cl = np.asarray(df_in["Close"], dtype=float).flatten()

  ha_close = (op + hi + lo + cl) / 4.0
  ha_open = np.zeros(len(df_in))
  ha_open[0] = (op[0] + cl[0]) / 2.0
  for i in range(1, len(df_in)):
    ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

  ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
  ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))

  df_out = df_in.copy()
  df_out["HA_Open"] = ha_open
  df_out["HA_High"] = ha_high
  df_out["HA_Low"] = ha_low
  df_out["HA_Close"] = ha_close
  return df_out


# -----------------------------------------------------------------
# 🛡️ BINANCE DIRECT 2-YEAR HOURLY FETCH (No Yahoo Dependency)
# -----------------------------------------------------------------
@st.cache_data(ttl=60)  # 1-Minute TTL Refresh
def fetch_binance_2year_hourly():
  url = "https://api.binance.com/api/v3/klines"
  symbol = "BTCUSDT"
  interval = "1h"
  limit = 1000

  # Calculate 2 years (730 days) back in milliseconds
  now_ms = int(time.time() * 1000)
  two_years_ms = 730 * 24 * 60 * 60 * 1000
  start_ms = now_ms - two_years_ms

  all_candles = []
  current_start = start_ms

  # Loop to fetch entire 2-year history (1000 candles per API call)
  while current_start < now_ms:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": current_start,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if not data or not isinstance(data, list):
      break

    all_candles.extend(data)
    # Move start time to last candle time + 1 ms
    current_start = data[-1][0] + 1

    # Safety stop if near current time
    if len(data) < limit:
      break

  # Convert to Pandas Dataframe
  cols = [
      "Open_Time",
      "Open",
      "High",
      "Low",
      "Close",
      "Volume",
      "Close_Time",
      "Quote_Vol",
      "Trades",
      "Taker_Buy_Base",
      "Taker_Buy_Quote",
      "Ignore",
  ]
  df_raw = pd.DataFrame(all_candles, columns=cols)

  # Convert numeric columns
  for col in ["Open", "High", "Low", "Close", "Volume"]:
    df_raw[col] = df_raw[col].astype(float)

  # Setup UTC Timestamp Index
  df_raw["Timestamp"] = pd.to_datetime(df_raw["Open_Time"], unit="ms", utc=True)
  df_raw.set_index("Timestamp", inplace=True)

  # 🔒 STRICT NON-LEAKAGE: Drop currently running/unclosed bar
  df_raw = df_raw.iloc[:-1]

  # Timezone conversion to IST
  df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")

  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


try:
  with st.spinner("Fetching Full 2-Year Hourly Binance Stream..."):
    df = fetch_binance_2year_hourly()
    if len(df) < 1000:
      st.error("🚨 Error: Insufficient data lines fetched from Binance.")
      st.stop()
except Exception as e:
  st.error(f"🚨 Binance API Failure: {e}")
  st.stop()

# =====================================================================
# ⚡ CORE TRANSFORMATIONS & DUAL KINEMATICS ENGINE
# =====================================================================
df = apply_heikin_ashi(df)

# Dynamic 50:50 Split Matrix across full 2-Year Engine
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Fetched {len(df)} Binance 1-Hour Candles (~2 Years) | Matrix"
    f" Processing {len(df_predict)} IST Locked Candles (Zero Leakage)!**"
)

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close = np.asarray(df_predict["Close"], dtype=float).flatten()
df_predict["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close, window=100
)
kalman_base_normal = apply_kalman_filter_custom(
    normal_close, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal = apply_kalman_filter_custom(
    normal_close - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_predict["HAM_Normal"] = momentum_normal * (
    df_predict["Hurst_Normal"].to_numpy() * 2.0
)

# --- PATH B: HEIKIN-ASHI CANDLE KINEMATICS ---
ha_close = np.asarray(df_predict["HA_Close"], dtype=float).flatten()
df_predict["Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close, window=100
)
kalman_base_ha = apply_kalman_filter_custom(
    ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha = apply_kalman_filter_custom(
    ha_close - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_predict["HAM_HeikinAshi"] = momentum_ha * (
    df_predict["Hurst_HA"].to_numpy() * 2.0
)

df_predict.dropna(subset=["Hurst_Normal", "Hurst_HA"], inplace=True)

# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    "Close",
    "HA_Close",
    "Hurst_Normal",
    "Hurst_HA",
    "HAM_Normal",
    "HAM_HeikinAshi",
]
display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
  display_df[col] = (
      np.asarray(df_predict[col], dtype=float).flatten().round(2)
  )

# Latest locked candle on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# 🎯 LATEST LOCKED CANDLE METRIC CARD
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDE (BINANCE IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,}")
col2.metric("Locked HA Close", f"${latest_candle['HA_Close']:,}")
col3.metric("Normal HAM Signal", f"{latest_candle['HAM_Normal']}")
col4.metric("HA HAM Signal", f"{latest_candle['HAM_HeikinAshi']}")

st.divider()

st.subheader("📋 2-Year 50:50 Dynamic Rolling Matrix (Binance Live Stream)")

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price ($)", format="$%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close ($)", format="$%.2f"
        ),
        "Hurst_Normal": st.column_config.NumberColumn(
            "Hurst (Normal)", format="%.2f"
        ),
        "Hurst_HA": st.column_config.NumberColumn("Hurst (HA)", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn(
            "HAM Normal Signal", format="%.2f"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA Signal", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
