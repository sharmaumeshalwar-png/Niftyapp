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
    page_title="BTC Kinematics State-Machine Engine", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) Dynamic Peak/Trough Hysteresis Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Continuous HAM Kinematics | **State-Machine Lock (Peak-to-Trough Trapping)** | Zero Flickering IST Matrix"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **State Lock Engine:** ACTIVE\n\n🎯 **Diff Kalman Q:** 0.0001"
)


# =====================================================================
# MATHEMATICAL ENGINES & STATE MACHINE
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.0005, r_val=0.2):
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


# STEP 7: PEAK/TROUGH HYSTERESIS STATE MACHINE (NO-FLICKER ENGINE)
def apply_hysteresis_state_machine(df_in, reversal_threshold_pct=0.20):
    df = df_in.copy()

    # Calculate Velocity & Acceleration using Smoothed HAM_Diff_Kalman
    df["HAM_Velocity"] = df["HAM_Diff_Kalman"].diff().fillna(0.0)
    df["HAM_Acceleration"] = df["HAM_Velocity"].diff().fillna(0.0)

    diff_vals = df["HAM_Diff_Kalman"].to_numpy()
    states = []

    # State tracking variables
    current_state = "🟡 INITIALIZING"
    peak_val = diff_vals[0]
    trough_val = diff_vals[0]

    for i in range(len(diff_vals)):
        val = diff_vals[i]

        if i == 0:
            states.append("🟡 INITIALIZING")
            continue

        # 1. Update Dynamic Peak & Trough Memory
        if val > peak_val:
            peak_val = val
        if val < trough_val:
            trough_val = val

        # Dynamic Threshold Range
        peak_drop_trigger = peak_val - (
            abs(peak_val) * reversal_threshold_pct + 1.0
        )
        trough_rise_trigger = trough_val + (
            abs(trough_val) * reversal_threshold_pct + 1.0
        )

        # 2. State Transition Logic
        if current_state in ["🟡 INITIALIZING", "🟢 STRONG BULLISH TREND"]:
            # Drop from peak check -> Lock into Bearish
            if val < peak_drop_trigger:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"
                trough_val = val  # Reset trough tracker for new down-cycle
            else:
                current_state = "🟢 STRONG BULLISH TREND"

        elif current_state == "🔴 STRONG BEARISH TREND (Rally Stopped)":
            # Surge from bottom check -> Lock into Bullish
            if val > trough_rise_trigger:
                current_state = "🟢 STRONG BULLISH TREND"
                peak_val = val  # Reset peak tracker for new up-cycle
            else:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"

        states.append(current_state)

    df["Flip_Status"] = states
    return df


# =====================================================================
# DUAL-SOURCE DATA FETCH ENGINE
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

    raise ValueError(
        "Both primary and fallback endpoints failed to return sufficient"
        " candles."
    )


# Fetch Data
try:
    with st.spinner(
        "🔄 Fetching Data & Applying Dynamic State Lock Engine..."
    ):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]  # Drop running candle
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# FULL CONTINUOUS KINEMATICS & STATE MACHINE
# =====================================================================
df = apply_heikin_ashi(df)

# Normal Path
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

# HA Path
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)
kalman_base_ha = apply_kalman_filter_custom(
    ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha = apply_kalman_filter_custom(
    ha_close_full - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

# Raw HAM Diff
df["HAM_Diff_Raw"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]

# AAPKI REQUIREMENT: Custom Kalman Filter applied on HAM Diff with q=0.0001
df["HAM_Diff_Kalman"] = apply_kalman_filter_custom(
    df["HAM_Diff_Raw"].to_numpy(), initial_p=0.50, q_val=0.0001, r_val=0.1
)

# Apply Peak/Trough Hysteresis Lock using Smoothed HAM_Diff_Kalman
df = apply_hysteresis_state_machine(df, reversal_threshold_pct=0.20)


# =====================================================================
# DISPLAY MATRIX (IST)
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

clean_cols = [
    "Close",
    "HA_Close",
    "Hurst_Normal",
    "Hurst_HA",
    "HAM_Normal",
    "HAM_HeikinAshi",
    "HAM_Diff_Kalman",
    "HAM_Velocity",
    "HAM_Acceleration",
    "Flip_Status",
]
display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    if col != "Flip_Status":
        display_df[col] = (
            np.asarray(df_predict[col], dtype=float).flatten().round(2)
        )
    else:
        display_df[col] = df_predict[col]

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Base HAM Normal", f"{latest_candle['HAM_Normal']:.2f}")
col3.metric("HAM HA Signal", f"{latest_candle['HAM_HeikinAshi']:.2f}")
col4.metric("📊 HAM Diff (Kalman)", f"{latest_candle['HAM_Diff_Kalman']:.2f}")
col5.metric("🎯 State Machine Status", f"{latest_candle['Flip_Status']}")

st.divider()

st.subheader(
    f"📋 Hysteresis Locked Kinematic Matrix ({len(display_df):,} Predict"
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
        "Hurst_Normal": st.column_config.NumberColumn(
            "Hurst (Normal)", format="%.2f"
        ),
        "Hurst_HA": st.column_config.NumberColumn(
            "Hurst (HA)", format="%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "Base HAM Normal", format="%.2f"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA Signal", format="%.2f"
        ),
        "HAM_Diff_Kalman": st.column_config.NumberColumn(
            "📊 HAM Diff (Kalman)", format="%.2f"
        ),
        "HAM_Velocity": st.column_config.NumberColumn(
            "⚡ Velocity (Δ1)", format="%.2f"
        ),
        "HAM_Acceleration": st.column_config.NumberColumn(
            "🚀 Acceleration (Δ2)", format="%.2f"
        ),
        "Flip_Status": st.column_config.TextColumn("🎯 Hysteresis Trend Lock"),
    },
    use_container_width=True,
    height=600,
)
