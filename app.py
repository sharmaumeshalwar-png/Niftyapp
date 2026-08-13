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
    page_title="BTC 2-Year Kinematics Engine (Kalman 0.10 Path)",
    layout="wide",
)
st.title("⚡ Bitcoin (BTC-USD) Kalman 0.10 Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** A, B (Kalman P0=0.10), C, D, E Matrix | 50:50 Split | IST Locked [Strict Zero Leakage]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Dual REST Stream:** CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================
def apply_kalman_filter_p010(data_array, initial_p=0.10, q_val=0.001, r_val=0.1):
    """
    Sequential single-pass Kalman Filter (Zero Leakage / No Repainting).
    Calculates B = Kalman Filter of A with P0 = 0.10 (Responsive setup).
    """
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
    """
    Calculates Hurst Exponent (D = Hurst of A).
    Vectorized Trailing R/S Hurst Exponent (30-Window Strict Causal).
    """
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
# DUAL-SOURCE DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
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

    raise ValueError(
        "Both primary and fallback endpoints failed to return sufficient candles."
    )


# Fetch Data
try:
    with st.spinner("🔄 Fetching 2 Years of Hourly BTC Data (~17,500 Candles)..."):
        df, source_used = get_robust_2year_hourly()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # 🔒 STRICT LEAKAGE PREVENTION: Drop unclosed running candle
        df = df.iloc[:-1]

        # Convert to IST
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ EXACT FORMULA KINEMATIC COMPUTATION (KALMAN P0 = 0.10)
# =====================================================================
# A = Close Normal
df["A_Close_Normal"] = np.asarray(df["Close"], dtype=float).flatten()

# B = Kalman of 0.10 of A
df["B_Kalman_0.10"] = apply_kalman_filter_p010(
    df["A_Close_Normal"].to_numpy(), initial_p=0.10
)

# C = A - B
df["C_Diff_Residual"] = df["A_Close_Normal"] - df["B_Kalman_0.10"]

# D = Hurst of value A
df["D_Hurst_A"] = calculate_rolling_hurst_vectorized(
    df["A_Close_Normal"].to_numpy(), window=30
)

# E = C * D
df["E_Kinematic_Signal"] = df["C_Diff_Residual"] * df["D_Hurst_A"]


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_predict.dropna(
    subset=["D_Hurst_A"],
    inplace=True,
)

st.success(
    f"🟢 **Synced via {source_used}: {total_candles:,} Total Candles** | 🧠"
    f" **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:**"
    f" {len(df_predict):,} (IST Locked)"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    "A_Close_Normal",
    "B_Kalman_0.10",
    "C_Diff_Residual",
    "D_Hurst_A",
    "E_Kinematic_Signal",
]
display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    display_df[col] = (
        np.asarray(df_predict[col], dtype=float).flatten().round(2)
    )

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# Metric Cards Display
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("A (Close Normal)", f"${latest_candle['A_Close_Normal']:,.2f}")
col2.metric("B (Kalman P0=0.10)", f"${latest_candle['B_Kalman_0.10']:,.2f}")
col3.metric("C (A - B)", f"{latest_candle['C_Diff_Residual']:.2f}")
col4.metric("D (Hurst of A)", f"{latest_candle['D_Hurst_A']:.2f}")
col5.metric("🔥 E (C * D)", f"{latest_candle['E_Kinematic_Signal']:.2f}")

st.divider()

st.subheader(
    f"📋 Kalman 0.10 Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "A_Close_Normal": st.column_config.NumberColumn(
            "A: Close Normal ($)", format="$%.2f"
        ),
        "B_Kalman_0.10": st.column_config.NumberColumn(
            "B: Kalman 0.10 ($)", format="$%.2f"
        ),
        "C_Diff_Residual": st.column_config.NumberColumn(
            "C: (A - B)", format="%.2f"
        ),
        "D_Hurst_A": st.column_config.NumberColumn(
            "D: Hurst(A)", format="%.2f"
        ),
        "E_Kinematic_Signal": st.column_config.NumberColumn(
            "🔥 E: (C * D)", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
