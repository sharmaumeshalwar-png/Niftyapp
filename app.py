from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="BTC Kinematics & Open Interest Engine", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) Kinematics & Futures Open Interest Engine")
st.write(
    "🎯 **1-Hour Timeframe:** HAM Normal (Kalman) | Futures Open Interest"
    " (Kalman) | OI / HAM Normal Ratio"
)

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
    """Standard Kalman Filter Engine for Signals."""
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
# DATA FETCH ENGINES (PRICE & OPEN INTEREST)
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_binance_klines(start_ts, end_ts):
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    all_candles = []
    for endpoint in endpoints:
        try:
            current_start = start_ts
            all_candles = []
            while current_start < end_ts:
                params = {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "startTime": current_start,
                    "limit": 1000,
                }
                res = requests.get(
                    endpoint, params=params, headers=headers, timeout=15
                )

                if res.status_code != 200:
                    break

                data = res.json()
                if not isinstance(data, list) or len(data) == 0:
                    break

                all_candles.extend(data)
                last_candle_time = data[-1][0]
                if last_candle_time <= current_start:
                    break
                current_start = last_candle_time + 1
                time.sleep(0.02)

            if len(all_candles) >= 300:
                break
        except Exception:
            continue

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
    df_raw["Timestamp"] = pd.to_datetime(
        df_raw["OpenTime"], unit="ms", utc=True
    )
    df_raw.set_index("Timestamp", inplace=True)
    return df_raw[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=1800)
def fetch_binance_open_interest():
    """Fetch BTC Futures Open Interest Data from Binance."""
    endpoint = "https://fapi.binance.com/futures/data/openInterestHist"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    params = {"symbol": "BTCUSDT", "period": "1h", "limit": 500}
    try:
        res = requests.get(
            endpoint, params=params, headers=headers, timeout=15
        ).json()
        if isinstance(res, list) and len(res) > 0:
            df_oi = pd.DataFrame(res)
            df_oi["sumOpenInterest"] = df_oi["sumOpenInterest"].astype(float)
            df_oi["sumOpenInterestValue"] = df_oi[
                "sumOpenInterestValue"
            ].astype(float)
            df_oi["Timestamp"] = pd.to_datetime(
                df_oi["timestamp"], unit="ms", utc=True
            )
            df_oi.set_index("Timestamp", inplace=True)
            return df_oi[["sumOpenInterest", "sumOpenInterestValue"]]
    except Exception:
        pass
    return None


# Fetch Data Execution
try:
    with st.spinner(
        "🔄 Fetching Price Klines & Futures Open Interest Data..."
    ):
        now = datetime.now(timezone.utc)
        start_dt = now - timedelta(days=30)
        df = fetch_binance_klines(
            int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
        )

        if df is None:
            st.error(
                "🚨 Price data fetch failed. Please click 'Force Refresh"
                " Engine' in sidebar."
            )
            st.stop()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Merge Open Interest
        df_oi = fetch_binance_open_interest()
        if df_oi is not None:
            df = df.join(df_oi, how="left")
            df["sumOpenInterest"] = (
                df["sumOpenInterest"].ffill().bfill().fillna(0.0)
            )
            df["sumOpenInterestValue"] = (
                df["sumOpenInterestValue"].ffill().bfill().fillna(0.0)
            )
        else:
            df["sumOpenInterest"] = 0.0
            df["sumOpenInterestValue"] = 0.0

        df = df.iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Engine Processing Error: {e}")
    st.stop()


# =====================================================================
# CALCULATIONS
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

# 2. Open Interest Kalman Filter Engine
oi_array = np.asarray(df["sumOpenInterest"], dtype=float).flatten()
df["OI_Kalman"] = apply_kalman_filter_custom(
    oi_array, initial_p=1000.0, q_val=0.01, r_val=1.0
)

# 3. BTC Futures OI Kalman / HAM Normal (Safe Division)
df["OI_HAM_Ratio"] = np.where(
    df["HAM_Normal"] != 0, df["OI_Kalman"] / df["HAM_Normal"], 0.0
)


# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

clean_cols = ["Close", "HAM_Normal", "OI_Kalman", "OI_HAM_Ratio"]

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
col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("HAM Normal", f"{latest_candle['HAM_Normal']:.4f}")
col3.metric("BTC Futures OI (Kalman)", f"{latest_candle['OI_Kalman']:,.2f} BTC")
col4.metric("OI / HAM Normal Ratio", f"{latest_candle['OI_HAM_Ratio']:,.2f}")

st.divider()

# Interactive Data Frame
st.subheader(
    f"📋 Kinematics & Open Interest Matrix ({len(display_df):,} Locked"
    " Candles)"
)
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price ($)", format="$%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "1. HAM Normal (Kalman)", format="%.4f"
        ),
        "OI_Kalman": st.column_config.NumberColumn(
            "2. BTC Futures OI (Kalman)", format="%.2f"
        ),
        "OI_HAM_Ratio": st.column_config.NumberColumn(
            "3. Futures OI / HAM Normal", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
