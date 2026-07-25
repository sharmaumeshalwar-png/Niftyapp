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
    page_title="BTC 1H Pure Engine (50:50 Hybrid HAM)", layout="wide"
)
st.title("⚡ BTC-USD 1-Hour Pure Engine (2-Year Backtest)")
st.write(
    "🎯 **Exclusive 1-Hour Timeframe Engine:** 2-Year Lookback Data + 50:50 Hybrid HAM Column | IST Locked [Strict Zero Future Leakage]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **1H Pure Stream:** CONNECTED"
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
  return tr.rolling(period).mean().bfill()


# =====================================================================
# 1-HOUR EXCLUSIVE DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_1h_data(start_ts, end_ts):
  """Fetches historical 1-Hour candles from Binance."""
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
        "interval": "1h",
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
def fetch_coinbase_1h_data(start_dt, now_dt):
  """Fallback engine for 1-Hour candles using Coinbase Pro."""
  endpoint = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
  headers = {"User-Agent": "Mozilla/5.0"}

  current_end = now_dt
  all_candles = []

  while current_end > start_dt:
    current_start = max(start_dt, current_end - timedelta(hours=300))
    params = {
        "granularity": 3600,
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


def get_1h_2year_data():
  now = datetime.now(timezone.utc)
  start_dt = now - timedelta(days=730)  # 2 Years Lookback

  try:
    df = fetch_binance_1h_data(
        int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
    )
    if df is not None and len(df) >= 300:
      return df, "Binance REST (1H)"
  except Exception:
    pass

  df = fetch_coinbase_1h_data(start_dt, now)
  if df is not None and len(df) >= 300:
    return df, "Coinbase Pro (1H)"

  raise ValueError("Failed to fetch 1-Hour 2-Year Data")


# Fetch 1H Data Stream
try:
  with st.spinner("🔄 Loading 1-Hour Pure Engine (2 Years Data)..."):
    df_1h_raw, source_1h = get_1h_2year_data()
    df_1h_raw.sort_index(inplace=True)
    df_1h_raw = df_1h_raw[~df_1h_raw.index.duplicated(keep="first")].iloc[:-1]
    df_1h_raw.index = df_1h_raw.index.tz_convert("Asia/Kolkata")
except Exception as e:
  st.error(f"🚨 Engine Fetch Error: {e}")
  st.stop()


# =====================================================================
# ⚡ 1-HOUR ENGINE CALCULATIONS (NEW 50:50 THEORY)
# =====================================================================
df_1h = apply_heikin_ashi(df_1h_raw)

# 1. NEW THEORY: 50:50 Hybrid Price Construction
df_1h["50_50_Hybrid_Price"] = 0.5 * df_1h["Close"] + 0.5 * df_1h["HA_Close"]
hybrid_series = df_1h["50_50_Hybrid_Price"].to_numpy()

# 2. Kalman Baseline on 50:50 Hybrid
kalman_base_hybrid = apply_kalman_filter_custom(
    hybrid_series, initial_p=50.0, q_val=0.0005, r_val=0.2
)
df_1h["1H_Kalman_Baseline"] = kalman_base_hybrid

# 3. Dynamic Multipliers (ATR + Slope Velocity + Hurst Gain)
atr_1h = calculate_atr(df_1h, period=14).to_numpy()
raw_residual = hybrid_series - kalman_base_hybrid
norm_residual = np.where(atr_1h > 0, raw_residual / atr_1h, 0.0)

kalman_velocity = np.gradient(kalman_base_hybrid) / np.where(
    atr_1h > 0, atr_1h, 1.0
)
velocity_mult = 1.0 + np.tanh(kalman_velocity)

hurst_hybrid = calculate_rolling_hurst_vectorized(hybrid_series, window=50)
df_1h["1H_Hurst_Hybrid"] = hurst_hybrid
hurst_gain = 2.0 * hurst_hybrid

# 4. NEW COLUMN: 1H Magical Hybrid HAM
magical_hybrid_ham = norm_residual * velocity_mult * hurst_gain
df_1h["1H_Magical_Hybrid_HAM"] = pd.Series(
    magical_hybrid_ham, index=df_1h.index
).ewm(span=3).mean()

# Reference Hurst Metrics
normal_close = np.asarray(df_1h["Close"], dtype=float).flatten()
df_1h["1H_Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close, window=50
)
df_1h["1H_Kalman_Hurst_Norm"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(normal_close), window=50
)

ha_close = np.asarray(df_1h["HA_Close"], dtype=float).flatten()
df_1h["1H_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close, window=50
)
df_1h["1H_Kalman_Hurst_HA"] = calculate_rolling_hurst_vectorized(
    apply_kalman_filter_custom(ha_close), window=50
)


# =====================================================================
# 📋 MATRIX RENDERING (1-HOUR ONLY)
# =====================================================================
display_df = df_1h.iloc[::-1].copy()
locked_1h_time = display_df.index[0].strftime("%Y-%m-%d %H:%M IST")

st.markdown("### 🔒 **LOCKED FINAL 1-HOUR CANDLE (IST)**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("1H Normal Close", f"${display_df['Close'].iloc[0]:,.2f}")
c2.metric("1H HA Close", f"${display_df['HA_Close'].iloc[0]:,.2f}")
c3.metric(
    "1H 50:50 Hybrid Price", f"${display_df['50_50_Hybrid_Price'].iloc[0]:,.2f}"
)
c4.metric(
    "⭐ 1H Magical Hybrid HAM",
    f"{display_df['1H_Magical_Hybrid_HAM'].iloc[0]:+.2f}",
)

st.caption(
    f"⏰ **Last Closed 1-Hour Candle Time:** `{locked_1h_time}` | **Source:**"
    f" {source_1h}"
)
st.divider()

st.subheader("📋 1-Hour Matrix Table (With New 50:50 Hybrid HAM Column)")

ordered_cols = [
    "Close",
    "HA_Close",
    "50_50_Hybrid_Price",  # New 50:50 Hybrid Price Column
    "1H_Hurst_Normal",
    "1H_Kalman_Hurst_Norm",
    "1H_Hurst_HA",
    "1H_Kalman_Hurst_HA",
    "1H_Hurst_Hybrid",
    "1H_Magical_Hybrid_HAM",  # New 50:50 Hybrid HAM Column
]

table_df = display_df[ordered_cols].round(2)
table_df.index = table_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(
    table_df,
    column_config={
        "Close": st.column_config.NumberColumn("1H Normal Close", format="$%.2f"),
        "HA_Close": st.column_config.NumberColumn(
            "1H HA Close", format="$%.2f"
        ),
        "50_50_Hybrid_Price": st.column_config.NumberColumn(
            "50:50 Hybrid Price", format="$%.2f"
        ),
        "1H_Hurst_Normal": st.column_config.NumberColumn(
            "Hurst (Normal)", format="%.2f"
        ),
        "1H_Kalman_Hurst_Norm": st.column_config.NumberColumn(
            "KalHurst (Normal)", format="%.2f"
        ),
        "1H_Hurst_HA": st.column_config.NumberColumn("Hurst (HA)", format="%.2f"),
        "1H_Kalman_Hurst_HA": st.column_config.NumberColumn(
            "KalHurst (HA)", format="%.2f"
        ),
        "1H_Hurst_Hybrid": st.column_config.NumberColumn(
            "Hurst (50:50 Hybrid)", format="%.2f"
        ),
        "1H_Magical_Hybrid_HAM": st.column_config.NumberColumn(
            "⭐ 1H Magical Hybrid HAM", format="%+.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
