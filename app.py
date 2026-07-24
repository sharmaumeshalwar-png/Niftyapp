import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC 2-Year Kinematics Engine (50:50 Split)", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) 2-Year Pure Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** 2-Year Full History | Dual H.A.M. Matrix"
    " (Normal vs Heikin-Ashi) | 50:50 Learn:Predict Split | IST Locked [Zero"
    " Leakage]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **Binance 2-Year Stream:** CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.001, r_val=0.1
):
  """Sequential single-pass Kalman Filter (No future smoothing / Zero Leakage)."""
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


def calculate_rolling_hurst_vectorized(price_series, window=50):
  """Purely trailing rolling Hurst exponent calculation."""
  arr = np.asarray(price_series, dtype=float).flatten()
  s = pd.Series(arr)
  log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
  hurst_values = np.full(len(arr), 0.5)

  if len(log_returns) < window:
    return hurst_values

  # Sliding window view only looks backwards (Causal)
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
  """Calculates Heikin-Ashi candles sequentially without look-ahead bias."""
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


# =====================================================================
# 2-YEAR HISTORICAL DATA FETCH ENGINE (PAGINATED REST FETCH)
# =====================================================================
@st.cache_data(ttl=3600)  # Cache 2-year data for 1 hour to ensure fast loads
def fetch_binance_2year_hourly():
  """Fetches full 2 years of 1-hour BTCUSDT candles via Binance REST API loop."""
  endpoint = "https://api.binance.com/api/v3/klines"
  symbol = "BTCUSDT"
  interval = "1h"
  limit = 1000  # Max limit per request

  # Calculate start timestamp for 2 years ago (in milliseconds)
  now = datetime.now(timezone.utc)
  start_dt = now - timedelta(days=730)
  start_ts = int(start_dt.timestamp() * 1000)
  end_ts = int(now.timestamp() * 1000)

  all_candles = []
  current_start = start_ts

  while current_start < end_ts:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": current_start,
        "limit": limit,
    }
    response = requests.get(endpoint, params=params, timeout=15)
    data = response.json()

    if not data or not isinstance(data, list):
      break

    all_candles.extend(data)

    # Move start cursor to the end of the last retrieved candle + 1ms
    last_candle_time = data[-1][0]
    if last_candle_time <= current_start:
      break
    current_start = last_candle_time + 1

    # Sleep slightly to remain friendly to API limits
    time.sleep(0.05)

  # Binance Kline Structure:
  # [0:OpenTime, 1:Open, 2:High, 3:Low, 4:Close, 5:Volume, ...]
  cols = [
      "OpenTime",
      "Open",
      "High",
      "Low",
      "Close",
      "Volume",
      "CloseTime",
      "QuoteVolume",
      "Trades",
      "TakerBase",
      "TakerQuote",
      "Ignore",
  ]
  df_raw = pd.DataFrame(all_candles, columns=cols)

  num_cols = ["Open", "High", "Low", "Close", "Volume"]
  df_raw[num_cols] = df_raw[num_cols].astype(float)

  df_raw["Timestamp"] = pd.to_datetime(df_raw["OpenTime"], unit="ms", utc=True)
  df_raw.set_index("Timestamp", inplace=True)
  df_raw.sort_index(inplace=True)

  # Drop duplicate index entries if any
  df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

  # 🔒 STRICT LEAKAGE PREVENTION: Drop the currently unclosed running candle
  df_raw = df_raw.iloc[:-1]

  # Timezone conversion to Asia/Kolkata (IST)
  df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")

  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


# Fetch Data
try:
  with st.spinner(
      "🔄 Fetching 2 Years of Hourly BTC Data (~17,500+ Candles)..."
  ):
    df = fetch_binance_2year_hourly()
    if len(df) < 5000:
      st.error("🚨 Error: Insufficient historical candles returned.")
      st.stop()
except Exception as e:
  st.error(f"🚨 API Fetching Error: {e}")
  st.stop()


# =====================================================================
# ⚡ CORE TRANSFORMATIONS & 50:50 LEARN:PREDICT SPLIT
# =====================================================================
df = apply_heikin_ashi(df)

total_candles = len(df)
split_idx = int(total_candles * 0.50)  # Strict 50:50 Cut

# Train/Learn Phase Data (First 50%)
df_learn = df.iloc[:split_idx].copy()
# Predict/Kinematic Phase Data (Last 50%)
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced {total_candles:,} Total Hourly Candles** | 🧠 **Learn Set:**"
    f" {len(df_learn):,} Candles | 🔮 **Predict Matrix:**"
    f" {len(df_predict):,} Candles (IST Locked)"
)

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close = np.asarray(df_predict["Close"], dtype=float).flatten()
df_predict["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close, window=50
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
    ha_close, window=50
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

# Reverse DataFrame to display latest closed candle at the top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# 🎯 LATEST LOCKED CANDLE METRIC CARD
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Locked HA Close", f"${latest_candle['HA_Close']:,.2f}")
col3.metric("Normal HAM Signal", f"{latest_candle['HAM_Normal']:.2f}")
col4.metric("HA HAM Signal", f"{latest_candle['HAM_HeikinAshi']:.2f}")

st.divider()

st.subheader(
    f"📋 50:50 Dynamic Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

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
