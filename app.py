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
    "🎯 **Direct Live Crypto Stream:** Full Multi-Level Kinematics (Normal vs"
    " Heikin-Ashi for High, Low, Close) in IST [2-Year Full Engine / Zero"
    " Leakage]"
)

# Sidebar Refresh Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE\n\n🔒 **Public Direct Stream:** CONNECTED"
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


def compute_kinematics_path(price_array, window=100):
  """Computes Independent Kalman, Hurst, and Scaled HAM for any Price Series"""
  arr = np.asarray(price_array, dtype=float).flatten()

  # 1. Base Kalman Trend
  kalman_vals = apply_kalman_filter_custom(
      arr, initial_p=50.0, q_val=0.0005, r_val=0.2
  )

  # 2. Rolling Hurst Exponent
  hurst_vals = calculate_rolling_hurst_vectorized(arr, window=window)

  # 3. Direct Scaled Momentum: (Price - Kalman) * Hurst Exponent
  price_diff = arr - kalman_vals
  ham_vals = price_diff * hurst_vals

  return kalman_vals, hurst_vals, ham_vals


def apply_heikin_ashi(df_in):
  """Generates Heikin-Ashi OHLC Values"""
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
# 🛡️ 2-YEAR UNRESTRICTED Crypto Endpoint (Kraken + Coinbase Fallback)
# -----------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_2year_public_crypto_hourly():
  url = "https://api.kraken.com/0/public/OHLC"
  params = {"pair": "XBTUSD", "interval": 60}

  response = requests.get(
      url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
  )
  data = response.json()

  if "result" not in data or not data["result"]:
    cb_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    cb_resp = requests.get(
        cb_url,
        params={"granularity": 3600},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    cb_data = cb_resp.json()
    cols = ["time", "Low", "High", "Open", "Close", "Volume"]
    df_raw = pd.DataFrame(cb_data, columns=cols)
    df_raw.sort_values(by="time", inplace=True)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["time"], unit="s", utc=True)
  else:
    pair_key = list(data["result"].keys())[0]
    raw_candles = data["result"][pair_key]
    cols = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
        "vwap",
        "Volume",
        "count",
    ]
    df_raw = pd.DataFrame(raw_candles, columns=cols)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
      df_raw[c] = df_raw[c].astype(float)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["time"], unit="s", utc=True)

  df_raw.set_index("Timestamp", inplace=True)

  # 🔒 STRICT NON-LEAKAGE: Drop currently running/unclosed bar
  df_raw = df_raw.iloc[:-1]

  # Timezone conversion to IST
  df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")

  return df_raw[["Open", "High", "Low", "Close", "Volume"]]


try:
  with st.spinner("Connecting to 2-Year Unrestricted Crypto Stream..."):
    df = fetch_2year_public_crypto_hourly()
    if len(df) < 50:
      st.error("🚨 Error: Insufficient data returned.")
      st.stop()
except Exception as e:
  st.error(f"🚨 API Connection Error: {e}")
  st.stop()

# =====================================================================
# ⚡ CORE TRANSFORMATIONS & DUAL KINEMATICS ENGINE
# =====================================================================
df = apply_heikin_ashi(df)

# Dynamic 50:50 Split Matrix
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced {len(df)} Live Hourly Candles | Matrix Processing"
    f" {len(df_predict)} IST Locked Candles (Zero Leakage Active)!**"
)

# -----------------------------------------------------------------
# 1️⃣ NORMAL OHLC KINEMATICS (Close, High, Low)
# -----------------------------------------------------------------
# Close
kalman_c, hurst_c, ham_c = compute_kinematics_path(
    df_predict["Close"], window=100
)
df_predict["Kalman_Close"] = kalman_c
df_predict["Hurst_Close"] = hurst_c
df_predict["HAM_Close"] = ham_c

# High
kalman_h, hurst_h, ham_h = compute_kinematics_path(
    df_predict["High"], window=100
)
df_predict["Kalman_High"] = kalman_h
df_predict["Hurst_High"] = hurst_h
df_predict["HAM_High"] = ham_h

# Low
kalman_l, hurst_l, ham_l = compute_kinematics_path(
    df_predict["Low"], window=100
)
df_predict["Kalman_Low"] = kalman_l
df_predict["Hurst_Low"] = hurst_l
df_predict["HAM_Low"] = ham_l

# -----------------------------------------------------------------
# 2️⃣ HEIKIN-ASHI KINEMATICS (HA_Close, HA_High, HA_Low)
# -----------------------------------------------------------------
# HA Close
kalman_ha_c, hurst_ha_c, ham_ha_c = compute_kinematics_path(
    df_predict["HA_Close"], window=100
)
df_predict["Kalman_HA_Close"] = kalman_ha_c
df_predict["Hurst_HA_Close"] = hurst_ha_c
df_predict["HAM_HA_Close"] = ham_ha_c

# HA High
kalman_ha_h, hurst_ha_h, ham_ha_h = compute_kinematics_path(
    df_predict["HA_High"], window=100
)
df_predict["Kalman_HA_High"] = kalman_ha_h
df_predict["Hurst_HA_High"] = hurst_ha_h
df_predict["HAM_HA_High"] = ham_ha_h

# HA Low
kalman_ha_l, hurst_ha_l, ham_ha_l = compute_kinematics_path(
    df_predict["HA_Low"], window=100
)
df_predict["Kalman_HA_Low"] = kalman_ha_l
df_predict["Hurst_HA_Low"] = hurst_ha_l
df_predict["HAM_HA_Low"] = ham_ha_l

df_predict.dropna(
    subset=["Hurst_Close", "Hurst_High", "Hurst_Low"], inplace=True
)

# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    "Close",
    "High",
    "Low",
    "HA_Close",
    "HA_High",
    "HA_Low",
    "Kalman_Close",
    "HAM_Close",
    "Kalman_High",
    "HAM_High",
    "Kalman_Low",
    "HAM_Low",
    "HAM_HA_Close",
    "HAM_HA_High",
    "HAM_HA_Low",
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

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,}")
col2.metric("High / Low HAM (Normal)", f"{latest_candle['HAM_High']} / {latest_candle['HAM_Low']}")
col3.metric("Locked HA Close", f"${latest_candle['HA_Close']:,}")
col4.metric("High / Low HAM (HA)", f"{latest_candle['HAM_HA_High']} / {latest_candle['HAM_HA_Low']}")

st.divider()

st.subheader("📋 Dynamic Kinematic Matrix (Normal vs Heikin-Ashi)")

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close ($)", format="$%.2f"),
        "High": st.column_config.NumberColumn("High ($)", format="$%.2f"),
        "Low": st.column_config.NumberColumn("Low ($)", format="$%.2f"),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close ($)", format="$%.2f"
        ),
        "HA_High": st.column_config.NumberColumn("HA High ($)", format="$%.2f"),
        "HA_Low": st.column_config.NumberColumn("HA Low ($)", format="$%.2f"),
        "Kalman_Close": st.column_config.NumberColumn(
            "Kalman (Close)", format="$%.2f"
        ),
        "HAM_Close": st.column_config.NumberColumn(
            "HAM (Close)", format="%.2f"
        ),
        "Kalman_High": st.column_config.NumberColumn(
            "Kalman (High)", format="$%.2f"
        ),
        "HAM_High": st.column_config.NumberColumn("HAM (High)", format="%.2f"),
        "Kalman_Low": st.column_config.NumberColumn(
            "Kalman (Low)", format="$%.2f"
        ),
        "HAM_Low": st.column_config.NumberColumn("HAM (Low)", format="%.2f"),
        "HAM_HA_Close": st.column_config.NumberColumn(
            "HAM (HA Close)", format="%.2f"
        ),
        "HAM_HA_High": st.column_config.NumberColumn(
            "HAM (HA High)", format="%.2f"
        ),
        "HAM_HA_Low": st.column_config.NumberColumn(
            "HAM (HA Low)", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
