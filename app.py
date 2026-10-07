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
st.title("⚡ Bitcoin (BTC-USD) Kinematics & Advanced Math Engine")
st.write(
    "🎯 **1-Hour Timeframe:** Kalman Filter + Custom Delta + Chaos/Ramsey/Phase-Space Features"
)

# Sidebar
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()


# =====================================================================
# MATHEMATICAL ENGINES (STRICTLY CAUSAL - NO FUTURE LEAK)
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=0.50, q_val=0.0001, r_val=0.1
):
    """Standard Kalman Filter Engine (Causal)."""
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
    """Rolling Hurst Exponent (Causal)."""
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


def calculate_chaos_divergence(price_series, window=30):
    """1. Chaos Theory Proxy: Local trajectory divergence in phase space.

    Measures sensitive dependence on past initial conditions without future
    leak.
    """
    arr = np.asarray(price_series, dtype=float).flatten()
    s = pd.Series(arr)
    returns = s.pct_change().fillna(0.0).to_numpy()
    chaos_vals = np.zeros(len(arr))

    if len(returns) < window:
        return chaos_vals

    # Rolling trajectory divergence proxy
    for i in range(window, len(returns)):
        sub_window = returns[i - window : i]
        diffs = np.abs(np.diff(sub_window))
        # Log mean divergence rate
        mean_diff = np.mean(diffs) + 1e-8
        max_diff = np.max(diffs) + 1e-8
        chaos_vals[i] = np.log(max_diff / mean_diff)

    return chaos_vals


def calculate_ramsey_order_ratio(price_series, window=30, pattern_len=4):
    """2. Ramsey Theory Metric: Measures structural order/patterns in random noise.

    Counts repeated binary movement combinations in past window.
    """
    arr = np.asarray(price_series, dtype=float).flatten()
    s = pd.Series(arr)
    binary_moves = (s.diff() > 0).astype(int).to_numpy()
    ramsey_vals = np.zeros(len(arr))

    if len(binary_moves) < window:
        return ramsey_vals

    for i in range(window, len(binary_moves)):
        sub_seq = binary_moves[i - window : i]
        # Count frequency of dominant 4-bit binary pattern
        patterns = [
            tuple(sub_seq[j : j + pattern_len])
            for j in range(len(sub_seq) - pattern_len + 1)
        ]
        if patterns:
            counts = pd.Series(patterns).value_counts()
            max_freq = counts.iloc[0]
            ramsey_vals[i] = max_freq / len(patterns)

    return ramsey_vals


def calculate_phase_space_distance(price_series, window=30, delay=2):
    """3. Takens' Theorem (Phase Space Reconstruction): 2D Delay Coordinates.

    Calculates current state distance from historical centroid in phase space.
    """
    arr = np.asarray(price_series, dtype=float).flatten()
    s = pd.Series(arr)
    log_ret = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    dist_vals = np.zeros(len(arr))

    if len(log_ret) < window + delay:
        return dist_vals

    for i in range(window + delay, len(log_ret)):
        sub_window = log_ret[i - window : i]
        x_pts = sub_window[:-delay]
        y_pts = sub_window[delay:]

        centroid_x = np.mean(x_pts)
        centroid_y = np.mean(y_pts)

        curr_x = log_ret[i - delay]
        curr_y = log_ret[i]

        # Euclidean distance in reconstructed phase space
        dist_vals[i] = np.sqrt(
            (curr_x - centroid_x) ** 2 + (curr_y - centroid_y) ** 2
        )

    return dist_vals


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
def fetch_coin
