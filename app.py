import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="BTC 1H Multi-HAM Matrix (Learn:Predict)", layout="wide"
)
st.title("⚡ BTC-USD 1-Hour Engine (Multi-HAM Matrix)")
st.write(
    "🎯 **50:50 Learn:Predict Architecture:** Pure Old HAM vs Old HAM vs New ATR"
    " Normalized HAM"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **50:50 Split Stream:** CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter(data_array, initial_p=50.0, q_val=0.0005, r_val=0.2):
  arr = np.asarray(data_array, dtype=float).flatten()
  if len(arr) == 0:
    return np.array([])
  x, p = arr[0], initial_p
  filtered = np.empty(len(arr))
  for i, z in enumerate(arr):
    p = p + q_val
    k = p / (p + r_val)
    x = x + k * (z - x)
    p = (1 - k) * p
    filtered[i] = x
  return filtered


def calculate_rolling_hurst(price_series, window=50):
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
  h_calc = np.full(len(rs_ratio), 0.5)
  h_calc[valid_mask] = np.log(rs_ratio[valid_mask]) / np.log(window)

  hurst_values[window - 1 :] = np.clip(h_calc, 0.0, 1.0)
  return hurst_values


def apply_heikin_ashi(df_in):
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
  high = df_in["High"]
  low = df_in["Low"]
  close = df_in["Close"].shift(1)
  tr = np.maximum(
      high - low, np.maximum(np.abs(high - close), np.abs(low - close))
  )
  return tr.rolling(period).mean().bfill()


# =====================================================================
# DUAL SOURCE DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_1h_2yr(start_ts, end_ts):
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
    try:
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
    except Exception:
      break

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
def fetch_coinbase_1h_2yr(start_dt, now_dt):
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
    try:
      res = requests.get(
          endpoint, params=params, headers=headers, timeout=10
      ).json()
      if isinstance(res, list) and len(res) > 0:
        all_candles.extend(res)
      else:
        break
      current_end = current_start
      time.sleep(0.05)
    except Exception:
      break

  if len(all_candles) < 500:
    return None

  cols = ["time", "Low", "High", "Open", "Close", "Volume"]
  df_raw = pd.DataFrame(all_candles, columns=cols)
  num_cols = ["Open", "High", "Low", "Close", "Volume"]
  df_raw[num_cols] = df_raw[num_cols].astype(float)
  df_raw["Timestamp"] = pd.to_datetime(df_raw["time"], unit="s", utc=True)
  df_raw.set_index("Timestamp", inplace=True)
  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


def get_robust_2year_data():
  now = datetime.now(timezone.utc)
  start_dt = now - timedelta(days=730)

  # Primary Source: Binance
  try:
    df = fetch_binance_1h_2yr(
        int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
    )
    if df is not None and len(df) >= 500:
      return df, "Binance REST API (1H)"
  except Exception:
    pass

  # Fallback Source: Coinbase Pro
  try:
    df = fetch_coinbase_1h_2yr(start_dt, now)
    if df is not None and len(df) >= 500:
      return df, "Coinbase Pro API (1H)"
  except Exception:
    pass

  return None, None


# Fetch Data
try:
  with st.spinner("🔄 Fetching 2-Year Dataset..."):
    df_raw, source = get_robust_2year_data()

    if df_raw is None:
      st.error(
          "🚨 Fetch Error: Unable to fetch data from Binance or Coinbase API."
          " Please click 'Force Refresh Engine' in sidebar."
      )
      st.stop()

    df_raw.sort_index(inplace=True)
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")].iloc[:-1]
    df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")
except Exception as e:
  st.error(f"🚨 Data Error: {e}")
  st.stop()

df_full = apply_heikin_ashi(df_raw)

# =====================================================================
# 50:50 LEARN vs PREDICT ENGINE
# =====================================================================
split_idx = len(df_full) // 2
df_learn = df_full.iloc[:split_idx].copy()
df_predict = df_full.iloc[split_idx:].copy()

# Noise Calibration for ATR HAM
learn_close = df_learn["Close"].to_numpy()
learn_kalman = apply_kalman_filter(learn_close)
learn_atr = calculate_atr(df_learn, period=14).to_numpy()
learn_residuals = (learn_close - learn_kalman) / np.where(
    learn_atr > 0, learn_atr, 1.0
)
calibrated_std = np.std(learn_residuals)

# Predict Phase Calculation
pred_close = df_predict["Close"].to_numpy()
pred_ha_close = df_predict["HA_Close"].to_numpy()

df_predict["Kalman_Close"] = apply_kalman_filter(pred_close)
df_predict["Kalman_HA"] = apply_kalman_filter(pred_ha_close)

df_predict["Hurst_Normal"] = calculate_rolling_hurst(pred_close, window=50)
df_predict["Hurst_HA"] = calculate_rolling_hurst(pred_ha_close, window=50)

pred_atr = calculate_atr(df_predict, period=14).to_numpy()
raw_res = pred_close - df_predict["Kalman_Close"].to_numpy()

# ---------------------------------------------------------------------
# 1. EXACT PURE OLD HAM SCORE: (Close - Kalman) * Hurst
# ---------------------------------------------------------------------
df_predict["🏷️ Pure_Old_HAM"] = raw_res * df_predict["Hurst_Normal"]

# ---------------------------------------------------------------------
# 2. OLD HAM SCORE (With Velocity Multiplier & 3-EMA)
# ---------------------------------------------------------------------
kalman_slope = np.gradient(df_predict["Kalman_Close"].to_numpy())
velocity_mult = 1.0 + np.tanh(
    kalman_slope / np.where(pred_atr > 0, pred_atr, 1.0)
)
hurst_gain = 2.0 * df_predict["Hurst_Normal"]

classic_ham = raw_res * velocity_mult * hurst_gain
# .ewm(span=3).mean() hi aapka 3-Period EMA hai
df_predict["⭐ Old_HAM_EMA3"] = (
    pd.Series(classic_ham, index=df_predict.index).ewm(span=3).mean()
)

# ---------------------------------------------------------------------
# 3. NEW ATR NORMALIZED HAM SCORE
# ---------------------------------------------------------------------
norm_res = raw_res / np.where(pred_atr > 0, pred_atr, 1.0)
atr_ham = (norm_res / (calibrated_std + 1e-6)) * df_predict["Hurst_Normal"]
df_predict["🛡️ ATR_Norm_HAM"] = (
    pd.Series(atr_ham, index=df_predict.index).ewm(span=3).mean()
)


# =====================================================================
# DASHBOARD MATRIX
# =====================================================================
latest_pred = df_predict.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Close", f"${latest_pred['Close']:,.2f}")
c2.metric("🏷️ Pure Old HAM", f"{latest_pred['🏷️ Pure_Old_HAM']:+.2f}")
c3.metric("⭐ Old HAM (3-EMA)", f"{latest_pred['⭐ Old_HAM_EMA3']:+.2f}")
c4.metric("🛡️ ATR Norm HAM", f"{latest_pred['🛡️ ATR_Norm_HAM']:+.2f}")

st.caption(
    f"⏰ **Last Candle Time:**"
    f" `{df_predict.index[-1].strftime('%Y-%m-%d %H:%M IST')}` | **Data"
    f" Source:** {source}"
)
st.divider()

st.subheader("📋 Predict Phase Matrix (3 HAM Variants)")

display_df = df_predict.iloc[::-1].copy()
cols_to_show = [
    "Close",
    "Kalman_Close",
    "Hurst_Normal",
    "🏷️ Pure_Old_HAM",  # Straight (Close - Kalman) * Hurst
    "⭐ Old_HAM_EMA3",  # Old HAM with Velocity + 3-EMA
    "🛡️ ATR_Norm_HAM",  # New Normalized HAM
]

table_df = display_df[cols_to_show].round(2)
table_df.index = table_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(
    table_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
        "Kalman_Close": st.column_config.NumberColumn(
            "Kalman (Close)", format="$%.2f"
        ),
        "Hurst_Normal": st.column_config.NumberColumn(
            "Hurst (Norm)", format="%.2f"
        ),
        "🏷️ Pure_Old_HAM": st.column_config.NumberColumn(
            "🏷️ Pure Old HAM", format="%+.2f"
        ),
        "⭐ Old_HAM_EMA3": st.column_config.NumberColumn(
            "⭐ Old HAM (3-EMA)", format="%+.2f"
        ),
        "🛡️ ATR_Norm_HAM": st.column_config.NumberColumn(
            "🛡️ New ATR HAM", format="%+.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
