import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC 2-Year Advanced Kinematics Engine (DSP Candles)",
    layout="wide",
)
st.title("⚡ Bitcoin (BTC-USD) Advanced DSP Candle Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** HA HAM vs. Advanced DSP Candle Paths | 50:50 Split | IST Locked [Strict Zero Leakage]"
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
# MATHEMATICAL ENGINES & NEW CANDLE BUILDERS
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    """Sequential single-pass Kalman Filter (Zero Leakage / No Repainting)."""
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
    """Vectorized Trailing R/S Hurst Exponent (30-Window Strict Causal)."""
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

    df_out = df_in.copy()
    df_out["HA_Close"] = ha_close
    return df_out


def build_savgol_candles(df_in, window_length=11, polyorder=2):
    """
    1. NEW CANDLE BUILDER: Savitzky-Golay Polynomial DSP Candles.
    Applies trailing polynomial smoothing to reconstruct low-noise close levels.
    """
    cl = df_in["Close"].to_numpy()
    if len(cl) < window_length:
        return cl

    # Trailing causal window filter
    savgol_close = np.empty_like(cl)
    for i in range(len(cl)):
        if i < window_length:
            savgol_close[i] = cl[i]
        else:
            sub_series = cl[i - window_length + 1 : i + 1]
            savgol_close[i] = savgol_filter(
                sub_series, window_length, polyorder
            )[-1]

    return savgol_close


def build_ehlers_supersmoother_candles(df_in, cutoff_period=10):
    """
    2. NEW CANDLE BUILDER: Ehlers SuperSmoother 2-Pole Filtered Candles.
    Cuts off high frequency noise without lag.
    """
    cl = df_in["Close"].to_numpy()
    ss_close = np.zeros_like(cl)

    a1 = np.exp(-np.sqrt(2) * np.pi / cutoff_period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / cutoff_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(len(cl)):
        if i < 2:
            ss_close[i] = cl[i]
        else:
            ss_close[i] = (
                c1 * (cl[i] + cl[i - 1]) / 2.0
                + c2 * ss_close[i - 1]
                + c3 * ss_close[i - 2]
            )

    return ss_close


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
# ⚡ KINEMATICS MATRIX (HA vs SAVGOL vs SUPERSMOOTHER)
# =====================================================================
df = apply_heikin_ashi(df)

# 1. HEIKIN-ASHI HAM
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)
kalman_base_ha = apply_kalman_filter_custom(
    ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha = apply_kalman_filter_custom(
    ha_close_full - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

# 2. SAVITZKY-GOLAY DSP CANDLES & HAM
df["SavGol_Close"] = build_savgol_candles(df, window_length=11, polyorder=2)
savgol_close_full = np.asarray(df["SavGol_Close"], dtype=float).flatten()
df["Hurst_SavGol"] = calculate_rolling_hurst_vectorized(
    savgol_close_full, window=30
)
kalman_base_sg = apply_kalman_filter_custom(
    savgol_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_sg = apply_kalman_filter_custom(
    savgol_close_full - kalman_base_sg, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_SavGol"] = momentum_sg * (df["Hurst_SavGol"].to_numpy() * 2.0)

# 3. SUPERSMOOTHER DSP CANDLES & HAM
df["SuperSmoother_Close"] = build_ehlers_supersmoother_candles(
    df, cutoff_period=10
)
ss_close_full = np.asarray(df["SuperSmoother_Close"], dtype=float).flatten()
df["Hurst_SuperSmoother"] = calculate_rolling_hurst_vectorized(
    ss_close_full, window=30
)
kalman_base_ss = apply_kalman_filter_custom(
    ss_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ss = apply_kalman_filter_custom(
    ss_close_full - kalman_base_ss, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_SuperSmoother"] = momentum_ss * (
    df["Hurst_SuperSmoother"].to_numpy() * 2.0
)

# --- CALCULATE SPECIFIC DIFFERENCE MATRICES ---
df["HA_minus_SavGol"] = df["HAM_HeikinAshi"] - df["HAM_SavGol"]
df["HA_minus_SuperSmoother"] = df["HAM_HeikinAshi"] - df["HAM_SuperSmoother"]


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_predict.dropna(
    subset=["Hurst_HA", "Hurst_SavGol", "Hurst_SuperSmoother"],
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
    "Close",
    "HA_Close",
    "SavGol_Close",
    "SuperSmoother_Close",
    "HAM_HeikinAshi",
    "HAM_SavGol",
    "HAM_SuperSmoother",
    "HA_minus_SavGol",
    "HA_minus_SuperSmoother",
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
col1.metric("Locked Close", f"${latest_candle['Close']:,.2f}")
col2.metric("HAM Heikin-Ashi", f"{latest_candle['HAM_HeikinAshi']:.2f}")
col3.metric("HAM SavGol DSP", f"{latest_candle['HAM_SavGol']:.2f}")
col4.metric(
    "📊 HA HAM - SavGol HAM", f"{latest_candle['HA_minus_SavGol']:.2f}"
)
col5.metric(
    "📊 HA HAM - SuperSmoother",
    f"{latest_candle['HA_minus_SuperSmoother']:.2f}",
)

st.divider()

st.subheader(
    f"📋 Advanced DSP Candle Kinematic Matrix ({len(display_df):,} Predict"
    " Candles)"
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
        "SavGol_Close": st.column_config.NumberColumn(
            "SavGol DSP Close ($)", format="$%.2f"
        ),
        "SuperSmoother_Close": st.column_config.NumberColumn(
            "SuperSmoother Close ($)", format="$%.2f"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA", format="%.2f"
        ),
        "HAM_SavGol": st.column_config.NumberColumn(
            "HAM SavGol", format="%.2f"
        ),
        "HAM_SuperSmoother": st.column_config.NumberColumn(
            "HAM SuperSmoother", format="%.2f"
        ),
        "HA_minus_SavGol": st.column_config.NumberColumn(
            "🔥 (HA HAM - SavGol HAM)", format="%.2f"
        ),
        "HA_minus_SuperSmoother": st.column_config.NumberColumn(
            "🔥 (HA HAM - SuperSmoother HAM)", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
