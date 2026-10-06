from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(page_title="BTC Kinematics Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Simplified Kinematics Engine")
st.write("🎯 **1-Hour Timeframe:** Normal HAM Kalman Core & Custom Delta Column")

# Sidebar
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=0.50, q_val=0.0001, r_val=0.1
):
    """Standard Kalman Filter Engine for Core HAM Signals."""
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


def calculate_rolling_hurst_vectorized(price_series, window=30):
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
    s_val = np.std(windows, axis=1, ddof=1) + 1e-10
    rs_ratio = r_val / s_val

    valid_mask = rs_ratio > 0
    h_calculated = np.full(len(rs_ratio), 0.5)
    h_calculated[valid_mask] = np.log(rs_ratio[valid_mask]) / np.log(window)

    hurst_values[window - 1 : window - 1 + len(h_calculated)] = np.clip(
        h_calculated, 0.0, 1.0
    )
    return hurst_values


# =====================================================================
# DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
    endpoint = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts
    headers = {"User-Agent": "Mozilla/5.0"}

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

    if len(all_candles) < 2000:
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
    df_raw["Timestamp"] = pd.to_datetime(
        df_raw["OpenTime"], unit="ms", utc=True
    )
    df_raw.set_index("Timestamp", inplace=True)
    return df_raw[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=3600)
def fetch_coinbase_data(start_dt, now_dt):
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
    df_raw.sort_index(ascending=True, inplace=True)
    return df_raw[["Open", "High", "Low", "Close", "Volume"]]


def get_robust_2year_hourly():
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=730)

    try:
        df = fetch_binance_data(
            int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
        )
        if df is not None and len(df) >= 5000:
            return df, "Binance REST API"
    except Exception:
        pass

    df = fetch_coinbase_data(start_dt, now)
    if df is not None and len(df) >= 2000:
        return df, "Coinbase Pro API (Fallback)"

    raise ValueError("Failed to fetch price data.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching Data & Calculating Engine..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# CALCULATION ENGINE
# =====================================================================
# 1. Base HAM Normal Signal
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)
kalman_base_normal = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)
df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

# 2. Custom Column Calculation: (Last Close - Current Close) + HAM_Normal
# Note: df['Close'].shift(1) gives the Last Candle's Close Price
df["HAM_Custom_Delta"] = (
    df["Close"].shift(1) - df["Close"]
) + df["HAM_Normal"]


# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

clean_cols = ["Close", "HAM_Normal", "HAM_Custom_Delta"]

display_df = pd.DataFrame(index=df_predict.index)
for col in clean_cols:
    display_df[col] = np.asarray(df_predict[col], dtype=float).flatten()

# Display latest candles at the top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

# Metrics Cards
col1, col2, col3 = st.columns(3)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("HAM Normal", f"{latest_candle['HAM_Normal']:.4f}")
col3.metric("Custom Delta Column", f"{latest_candle['HAM_Custom_Delta']:.4f}")

st.divider()

# Interactive Data Frame
st.subheader(f"📋 Custom Kinematic Matrix ({len(display_df):,} Locked Candles)")
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price ($)", format="$%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "HAM Normal (Kalman)", format="%.4f"
        ),
        "HAM_Custom_Delta": st.column_config.NumberColumn(
            "(Last Close - Current Close) + HAM Normal", format="%.4f"
        ),
    },
    use_container_width=True,
    height=600,
)
