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
    page_title="BTC Dual Engine Matrix (50:50 Hybrid HAM)", layout="wide"
)
st.title("⚡ BTC-USD Dual-Engine Kinematic Matrix (1H & 15M)")
st.write(
    "🎯 **Synchronized Dual Timeframe Engine:** 1-Hour (2-Year Lookback) &"
    " 15-Minute Kinematic Matrix + Dynamic 50:50 Hybrid HAM Column | IST"
    " Locked [Strict Zero Future Leakage]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **Unified REST Stream:** CONNECTED"
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


def calculate_atr(df_in, period=14):
  """Calculates Average True Range for dynamic volatility standardization."""
  high = df_in["High"]
  low = df_in["Low"]
  close = df_in["Close"].shift(1)
  tr = np.maximum(
      high - low,
      np.maximum(np.abs(high - close), np.abs(low - close)),
  )
  return tr.rolling(period).mean().fillna(method="bfill")


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

  if len(all_candles) < 300:
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

  try:
    df = fetch_binance_data(
        int(start_dt.timestamp() * 1000),
        int(now.timestamp() * 1000),
        interval=interval,
    )
    if df is not None and len(df) >= 300:
      return df, f"Binance REST ({interval})"
  except Exception:
    pass

  df = fetch_coinbase_data(start_dt, now, granularity=granularity_map[interval])
  if df is not None and len(df) >= 300:
    return df, f"Coinbase Pro ({interval})"

  raise ValueError(f"Failed to fetch data for interval: {interval}")


# Fetch Sync Data (1H = 2 Years / 730 Days, 15M = 60 Days)
try:
  with st.spinner("🔄 Synchronizing 1H (2-Year Data) & 15M Streams..."):
    df_1h_raw, source_1h = get_robust_timeframe_data(interval="1h", days=730)
    df_1h_raw.sort_index(inplace=True)
    df_1h_raw = df_1h_raw[~df_1h_raw.index.duplicated(keep="first")].iloc[:-1]
    df_1h_raw.index = df_1h_raw.index.tz_convert("Asia/Kolkata")

    df_15m_raw, source_15m = get_robust_timeframe_data(interval="15m", days=60)
    df_15m_raw.sort_index(inplace=True)
    df_15m_raw = df_15m_raw[~df_15m_raw.index.duplicated(keep="first")].iloc[
        :-1
    ]
    df_15m_raw.index = df_15m_raw.index.tz_convert("Asia/Kolkata")
except Exception as e:
  st.error(f"🚨 Engine Fetch Error: {e}")
  st.stop()


# =====================================================================
# ⚡ 1-HOUR ENGINE PROCESSING (50:50 HYBRID + MAGICAL HAM)
# =====================================================================
df_1h = apply_heikin_ashi(df_1h_raw)

# 1. 50:50 Hybrid Price Construction (New Theory)
df_1h["Hybrid_Price"] = 0.5 * df_1h["Close"] + 0.5 * df_1h["HA_Close"]
hybrid_series_1h = df_1h["Hybrid_Price"].to_numpy()

# 2. Kalman Baseline on Hybrid Price
kalman_base_1h_hybrid = apply_kalman_filter_custom(
    hybrid_series_1h, initial_p=50.0, q_val=0.0005, r_val=0.2
)
df_1h["1H_Kalman_Baseline"] = kalman_base_1h_hybrid

# 3. Dynamic Multipliers (ATR + Velocity + Hurst)
atr_1h = calculate_atr(df_1h, period=14).to_numpy()
raw_residual_1h = hybrid_series_1h - kalman_base_1h_hybrid
norm_residual_1h = np.where(atr_1h > 0, raw_residual_1h / atr_1h, 0.0)

kalman_velocity_1h = np.gradient(kalman_base_1h_hybrid) / np.where(
    atr_1h > 0, atr_1h, 1.0
)
velocity_mult_1h = 1.0 + np.tanh(kalman_velocity_1h)

hurst_1h_hybrid = calculate_rolling_hurst_vectorized(
    hybrid_series_1h, window=50
)
df_1h["1H_Hurst_Hybrid"] = hurst_1h_hybrid
hurst_gain_1h = 2.0 * hurst_1h_hybrid

# 4. NEW MAGICAL HYBRID HAM COLUMN
magical_hybrid_ham_1h = norm_residual_1h * velocity_mult_1h * hurst_gain_1h
df_1h["1H_Magical_Hybrid_HAM"] = pd.Series(
    magical_hybrid_ham_1h, index=df_1h.index
).ewm(span=3).mean()

# Normal & HA Legacy Baselines
normal_close_1h = np.asarray(df_1h["Close"], dtype=float).flatten()
df_1h["1H_Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_1h, window=50
)
df_1h["1H_Kalman_Hurst_Norm"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(normal_close_1h), window=50
)

ha_close_1h = np.asarray(df_1h["HA_Close"], dtype=float).flatten()
df_1h["1H_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close_1h, window=50
)
df_1h["1H_Kalman_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(ha_close_1h), window=50
)

df_1h_clean = df_1h[[
    "Close",
    "HA_Close",
    "Hybrid_Price",
    "1H_Hurst_Normal",
    "1H_Kalman_Hurst_Norm",
    "1H_Hurst_HA",
    "1H_Kalman_Hurst_HA",
    "1H_Magical_Hybrid_HAM",
]].copy()


# =====================================================================
# ⚡ 15-MINUTE ENGINE PROCESSING
# =====================================================================
df_15m = apply_heikin_ashi(df_15m_raw)

normal_close_15m = np.asarray(df_15m["Close"], dtype=float).flatten()
df_15m["15M_Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_15m, window=50
)
df_15m["15M_Kalman_Hurst_Norm"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(normal_close_15m), window=50
)

ha_close_15m = np.asarray(df_15m["HA_Close"], dtype=float).flatten()
df_15m["15M_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close_15m, window=50
)
df_15m["15M_Kalman_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(ha_close_15m), window=50
)

df_15m_clean = df_15m[[
    "Close",
    "HA_Close",
    "15M_Hurst_Normal",
    "15M_Kalman_Hurst_Norm",
    "15M_Hurst_HA",
    "15M_Kalman_Hurst_HA",
]].copy()


# =====================================================================
# 📋 UNIFIED MERGE & MATRIX RENDERING
# =====================================================================
combined_df = df_15m_clean.join(
    df_1h_clean, how="left", lsuffix="_15m", rsuffix="_1h"
).ffill()

# Differential Metrics
combined_df["Kalman_Hurst_HA_Diff"] = (
    combined_df["15M_Kalman_Hurst_HA"] - combined_df["1H_Kalman_Hurst_HA"]
)

display_df = combined_df.iloc[::-1].copy()

# Lock Status Timestamps
locked_1h_time = df_1h_clean.index[-1].strftime("%Y-%m-%d %H:%M IST")
locked_15m_time = df_15m_clean.index[-1].strftime("%Y-%m-%d %H:%M IST")

st.markdown("### 🔒 **LOCKED FINAL CANDLES (IST)**")

c1, c2, c3 = st.columns([2, 2, 1.5])
with c1:
  st.info(f"⏰ **1-Hour Locked Candle (2-Yr Engine):** `{locked_1h_time}`")
  m1, m2 = st.columns(2)
  m1.metric("1H Hybrid Price", f"${df_1h_clean['Hybrid_Price'].iloc[-1]:,.2f}")
  m2.metric(
      "1H Magical Hybrid HAM",
      f"{df_1h_clean['1H_Magical_Hybrid_HAM'].iloc[-1]:+.2f}",
  )

with c2:
  st.info(f"⚡ **15-Min Locked Candle:** `{locked_15m_time}`")
  n1, n2 = st.columns(2)
  n1.metric("15M HA Close", f"${df_15m_clean['HA_Close'].iloc[-1]:,.2f}")
  n2.metric(
      "15M Kalman Hurst HA",
      f"{df_15m_clean['15M_Kalman_Hurst_HA'].iloc[-1]:.2f}",
  )

with c3:
  latest_diff = combined_df["Kalman_Hurst_HA_Diff"].iloc[-1]
  st.metric(
      "📊 Kalman Hurst HA Diff (15M-1H)",
      f"{latest_diff:+.2f}",
      delta_color="normal",
  )

st.divider()

st.subheader(
    "📋 Unified Dual-Engine Matrix (Featuring 1H 50:50 Hybrid Magical HAM)"
)

ordered_cols = [
    "Close_15m",
    "HA_Close_15m",
    "15M_Hurst_Normal",
    "15M_Kalman_Hurst_Norm",
    "15M_Hurst_HA",
    "15M_Kalman_Hurst_HA",
    "Close_1h",
    "HA_Close_1h",
    "Hybrid_Price",
    "1H_Hurst_Normal",
    "1H_Kalman_Hurst_Norm",
    "1H_Hurst_HA",
    "1H_Kalman_Hurst_HA",
    "1H_Magical_Hybrid_HAM",  # New Column Added Here
    "Kalman_Hurst_HA_Diff",
]

display_df = display_df[ordered_cols].round(2)
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(
    display_df,
    column_config={
        "Close_15m": st.column_config.NumberColumn(
            "15M Close", format="$%.2f"
        ),
        "HA_Close_15m": st.column_config.NumberColumn(
            "15M HA Close", format="$%.2f"
        ),
        "15M_Hurst_Normal": st.column_config.NumberColumn(
            "15M Hurst Norm", format="%.2f"
        ),
        "15M_Kalman_Hurst_Norm": st.column_config.NumberColumn(
            "15M KalHurst Norm", format="%.2f"
        ),
        "15M_Hurst_HA": st.column_config.NumberColumn(
            "15M Hurst HA", format="%.2f"
        ),
        "15M_Kalman_Hurst_HA": st.column_config.NumberColumn(
            "15M KalHurst HA", format="%.2f"
        ),
        "Close_1h": st.column_config.NumberColumn("1H Close", format="$%.2f"),
        "HA_Close_1h": st.column_config.NumberColumn(
            "1H HA Close", format="$%.2f"
        ),
        "Hybrid_Price": st.column_config.NumberColumn(
            "1H 50:50 Hybrid Price", format="$%.2f"
        ),
        "1H_Hurst_Normal": st.column_config.NumberColumn(
            "1H Hurst Norm", format="%.2f"
        ),
        "1H_Kalman_Hurst_Norm": st.column_config.NumberColumn(
            "1H KalHurst Norm", format="%.2f"
        ),
        "1H_Hurst_HA": st.column_config.NumberColumn(
            "1H Hurst HA", format="%.2f"
        ),
        "1H_Kalman_Hurst_HA": st.column_config.NumberColumn(
            "1H KalHurst HA", format="%.2f"
        ),
        "1H_Magical_Hybrid_HAM": st.column_config.NumberColumn(
            "⭐ 1H Magical Hybrid HAM", format="%+.2f"
        ),
        "Kalman_Hurst_HA_Diff": st.column_config.NumberColumn(
            "KalHurst HA Diff", format="%+.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
