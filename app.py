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
    page_title="BTC Multi-Timeframe Kinematics Engine", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) Dual Timeframe (1H + 15M) Kinematic Engine")
st.write(
    "🎯 **Dual Timeframe Engine:** 1-Hour & 15-Minute Kinematic Matrix | "
    "Normal vs Heikin-Ashi | IST Locked [Strict Zero Future Leakage]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **Dual Stream (1H & 15M):** CONNECTED"
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
# DUAL TIMEFRAME DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts, interval="1h"):
  """Fetches historical candles from Binance for specified interval."""
  endpoint = "https://api.binance.com/api/v3/klines"
  all_candles = []
  current_start = start_ts

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  while current_start < end_ts:
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "startTime": current_start,
        "limit": 1000,
    }
    res = requests.get(
        endpoint, params=params, headers=headers, timeout=10
    ).json()

    if not isinstance(res, list) or len(res) == 0:
      break

    all_candles.extend(res)
    last_candle_time = res[-1][0]
    if last_candle_time <= current_start:
      break
    current_start = last_candle_time + 1
    time.sleep(0.02)

  if len(all_candles) < 500:
    return None

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
  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=3600)
def fetch_coinbase_data(start_dt, now_dt, granularity=3600):
  """Fallback engine using Coinbase Pro Pagination."""
  endpoint = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
  headers = {"User-Agent": "Mozilla/5.0"}

  current_end = now_dt
  all_candles = []

  # Calculate hours per query limit (max 300 candles per query)
  hours_step = (300 * granularity) // 3600

  while current_end > start_dt:
    current_start = max(start_dt, current_end - timedelta(hours=hours_step))
    params = {
        "granularity": granularity,
        "start": current_start.isoformat(),
        "end": current_end.isoformat(),
    }
    res = requests.get(
        endpoint, params=params, headers=headers, timeout=10
    ).json()

    if isinstance(res, list) and len(res) > 0:
      all_candles.extend(res)
    else:
      break

    current_end = current_start
    time.sleep(0.05)

  if len(all_candles) == 0:
    return None

  cols = ["time", "Low", "High", "Open", "Close", "Volume"]
  df_raw = pd.DataFrame(all_candles, columns=cols)
  num_cols = ["Open", "High", "Low", "Close", "Volume"]
  df_raw[num_cols] = df_raw[num_cols].astype(float)
  df_raw["Timestamp"] = pd.to_datetime(df_raw["time"], unit="s", utc=True)
  df_raw.set_index("Timestamp", inplace=True)
  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


def get_robust_timeframe_data(interval="1h", days=730):
  now = datetime.now(timezone.utc)
  start_dt = now - timedelta(days=days)

  granularity_map = {"1h": 3600, "15m": 900}

  # Try Primary Source (Binance)
  try:
    df = fetch_binance_data(
        int(start_dt.timestamp() * 1000),
        int(now.timestamp() * 1000),
        interval=interval,
    )
    if df is not None and len(df) >= 500:
      return df, f"Binance REST ({interval})"
  except Exception:
    pass

  # Fallback to Secondary Source (Coinbase Pro)
  df = fetch_coinbase_data(start_dt, now, granularity=granularity_map[interval])
  if df is not None and len(df) >= 500:
    return df, f"Coinbase Pro ({interval})"

  raise ValueError(f"Failed to fetch sufficient data for interval: {interval}")


# Fetch 1-Hour Data
try:
  with st.spinner("🔄 Fetching 1-Hour BTC Data..."):
    df_1h, source_1h = get_robust_timeframe_data(interval="1h", days=730)
    df_1h.sort_index(inplace=True)
    df_1h = df_1h[~df_1h.index.duplicated(keep="first")]

    # 🔒 STRICT LEAKAGE PREVENTION: Drop unclosed 1H running candle
    df_1h = df_1h.iloc[:-1]
    df_1h.index = df_1h.index.tz_convert("Asia/Kolkata")
except Exception as e:
  st.error(f"🚨 1-Hour Engine Error: {e}")
  st.stop()

# Fetch 15-Minute Data
try:
  with st.spinner("🔄 Fetching 15-Minute BTC Data..."):
    # Fetching last 60 days for 15-minute stream for high-density analysis
    df_15m, source_15m = get_robust_timeframe_data(interval="15m", days=60)
    df_15m.sort_index(inplace=True)
    df_15m = df_15m[~df_15m.index.duplicated(keep="first")]

    # 🔒 STRICT LEAKAGE PREVENTION: Drop unclosed 15M running candle
    df_15m = df_15m.iloc[:-1]
    df_15m.index = df_15m.index.tz_convert("Asia/Kolkata")
except Exception as e:
  st.error(f"🚨 15-Minute Engine Error: {e}")
  st.stop()


# =====================================================================
# ⚡ 1-HOUR ENGINE PROCESSING (EXISTING CODE - UNTOUCHED)
# =====================================================================
df_1h = apply_heikin_ashi(df_1h)

total_candles_1h = len(df_1h)
split_idx_1h = int(total_candles_1h * 0.50)

df_learn_1h = df_1h.iloc[:split_idx_1h].copy()
df_predict_1h = df_1h.iloc[split_idx_1h:].copy()

# 1H Path A: Normal
normal_close_1h = np.asarray(df_predict_1h["Close"], dtype=float).flatten()
df_predict_1h["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_1h, window=50
)
kalman_base_normal_1h = apply_kalman_filter_custom(
    normal_close_1h, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal_1h = apply_kalman_filter_custom(
    normal_close_1h - kalman_base_normal_1h,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)
df_predict_1h["HAM_Normal"] = momentum_normal_1h * (
    df_predict_1h["Hurst_Normal"].to_numpy() * 2.0
)

# 1H Path B: Heikin-Ashi
ha_close_1h = np.asarray(df_predict_1h["HA_Close"], dtype=float).flatten()
df_predict_1h["Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close_1h, window=50
)
kalman_base_ha_1h = apply_kalman_filter_custom(
    ha_close_1h, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha_1h = apply_kalman_filter_custom(
    ha_close_1h - kalman_base_ha_1h, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_predict_1h["HAM_HeikinAshi"] = momentum_ha_1h * (
    df_predict_1h["Hurst_HA"].to_numpy() * 2.0
)


# =====================================================================
# ⚡ 15-MINUTE KINEMATICS ENGINE ADDITION
# =====================================================================
df_15m = apply_heikin_ashi(df_15m)

# 15M Path A: Normal Close Kinematics
normal_close_15m = np.asarray(df_15m["Close"], dtype=float).flatten()
df_15m["15M_Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_15m, window=50
)
kalman_base_normal_15m = apply_kalman_filter_custom(
    normal_close_15m, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal_15m = apply_kalman_filter_custom(
    normal_close_15m - kalman_base_normal_15m,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)
df_15m["15M_HAM_Normal"] = momentum_normal_15m * (
    df_15m["15M_Hurst_Normal"].to_numpy() * 2.0
)

# 15M Path B: Heikin-Ashi Close Kinematics
ha_close_15m = np.asarray(df_15m["HA_Close"], dtype=float).flatten()
df_15m["15M_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close_15m, window=50
)
kalman_base_ha_15m = apply_kalman_filter_custom(
    ha_close_15m, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha_15m = apply_kalman_filter_custom(
    ha_close_15m - kalman_base_ha_15m, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_15m["15M_HAM_HeikinAshi"] = momentum_ha_15m * (
    df_15m["15M_Hurst_HA"].to_numpy() * 2.0
)


# =====================================================================
# 📋 DISPLAY & LOCKED CANDLE STATUS CARDS
# =====================================================================
st.success(
    f"🟢 **1H Stream:** {len(df_predict_1h):,} Predict Candles | 🟢 **15M Stream:**"
    f" {len(df_15m):,} Processing Candles (IST Locked)"
)

# Time Alignment for Display (Reversed Index)
df_1h_display = df_predict_1h.iloc[::-1].copy()
df_15m_display = df_15m.iloc[::-1].copy()

# Lock Status Timestamps
locked_1h_time = df_1h_display.index[0].strftime("%Y-%m-%d %H:%M IST")
locked_15m_time = df_15m_display.index[0].strftime("%Y-%m-%d %H:%M IST")

st.markdown("### 🔒 **LOCKED CANDLE TIME STATUS (IST)**")

c1, c2 = st.columns(2)
with c1:
  st.info(f"⏰ **1-Hour Final Locked Candle:** `{locked_1h_time}`")
  m1, m2, m3 = st.columns(3)
  m1.metric("1H Close", f"${df_1h_display['Close'].iloc[0]:,.2f}")
  m2.metric("1H HAM Normal", f"{df_1h_display['HAM_Normal'].iloc[0]:.2f}")
  m3.metric("1H HAM HA", f"{df_1h_display['HAM_HeikinAshi'].iloc[0]:.2f}")

with c2:
  st.info(f"⚡ **15-Min Final Locked Candle:** `{locked_15m_time}`")
  n1, n2, n3 = st.columns(3)
  n1.metric("15M Close", f"${df_15m_display['Close'].iloc[0]:,.2f}")
  n2.metric("15M HAM Normal", f"{df_15m_display['15M_HAM_Normal'].iloc[0]:.2f}")
  n3.metric("15M HAM HA", f"{df_15m_display['15M_HAM_HeikinAshi'].iloc[0]:.2f}")

st.divider()

st.subheader("📋 15-Minute Kinematic Signal Matrix")

cols_15m = [
    "Close",
    "HA_Close",
    "15M_Hurst_Normal",
    "15M_Hurst_HA",
    "15M_HAM_Normal",
    "15M_HAM_HeikinAshi",
]
display_15m_df = pd.DataFrame(index=df_15m_display.index)

for col in cols_15m:
  display_15m_df[col] = (
      np.asarray(df_15m_display[col], dtype=float).flatten().round(2)
  )

display_15m_df.index = display_15m_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(
    display_15m_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "15M Close ($)", format="$%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "15M HA Close ($)", format="$%.2f"
        ),
        "15M_Hurst_Normal": st.column_config.NumberColumn(
            "Hurst Normal (15M)", format="%.2f"
        ),
        "15M_Hurst_HA": st.column_config.NumberColumn(
            "Hurst HA (15M)", format="%.2f"
        ),
        "15M_HAM_Normal": st.column_config.NumberColumn(
            "15M HAM Normal Signal", format="%.2f"
        ),
        "15M_HAM_HeikinAshi": st.column_config.NumberColumn(
            "15M HAM HA Signal", format="%.2f"
        ),
    },
    use_container_width=True,
    height=500,
)
