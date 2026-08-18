import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC VWAP-Hurst Kinematics Engine", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) VWAP+Hurst Kinematics Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Continuous HAM Kinematics (Kalman Core) |"
    " **Adaptive VWAP + Hurst Hubble Basis** | **Gaussian Filtered Expansion**"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")

gaussian_sigma = st.sidebar.slider(
    "🔔 Hubble Gaussian Sigma (σ)",
    min_value=0.5,
    max_value=20.0,
    value=3.0,
    step=0.5,
    help="Gaussian bell-curve smoothing applied to Hubble Velocity.",
)

if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict 50:50 Train/Predict Split)\n\n"
    "🎯 **Hubble Basis Engine:** HYBRID VWAP DISPERSION + HURST ADAPTIVE\n\n"
    "🔒 **State Lock Engine:** ACTIVE\n\n"
    "⚡ **Base HAM Core:** KALMAN FILTER ACTIVE\n\n"
    f"🔔 **Hubble Expansion Filter:** GAUSSIAN (Sigma = {gaussian_sigma})"
)


# =====================================================================
# MATHEMATICAL ENGINES & FILTERS
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=0.50, q_val=0.0001, r_val=0.1, last_x=None, last_p=None
):
    """Kalman Filter Engine supporting state carryover across splits."""
    arr = np.asarray(data_array, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([]), initial_p, initial_p

    x = arr[0] if last_x is None else last_x
    p = initial_p if last_p is None else last_p

    filtered_values = np.empty(len(arr))
    for i, z in enumerate(arr):
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values[i] = x
    return filtered_values, x, p


def apply_gaussian_smoothing(data_array, sigma=3.0):
    arr = np.asarray(data_array, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([])
    return gaussian_filter1d(arr, sigma=sigma, mode="nearest")


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


def apply_hysteresis_state_machine(df_in, split_idx, reversal_threshold_pct=0.20):
    df = df_in.copy()

    raw_velocity = df["HAM_Diff_Kalman"].diff().fillna(0.0).to_numpy()
    vel_filtered, _, _ = apply_kalman_filter_custom(
        raw_velocity, initial_p=0.50, q_val=0.000001, r_val=0.1
    )
    df["HAM_Velocity"] = vel_filtered
    df["HAM_Acceleration"] = df["HAM_Velocity"].diff().fillna(0.0)

    diff_vals = df["HAM_Diff_Kalman"].to_numpy()
    states = []

    current_state = "🟡 INITIALIZING"
    peak_val = diff_vals[0]
    trough_val = diff_vals[0]

    for i in range(len(diff_vals)):
        val = diff_vals[i]

        if val > peak_val:
            peak_val = val
        if val < trough_val:
            trough_val = val

        peak_drop_trigger = peak_val - (
            abs(peak_val) * reversal_threshold_pct + 1.0
        )
        trough_rise_trigger = trough_val + (
            abs(trough_val) * reversal_threshold_pct + 1.0
        )

        phase_prefix = "🟢 [PREDICT]" if i >= split_idx else "📘 [LEARN]"

        if current_state in ["🟡 INITIALIZING", "🟢 STRONG BULLISH TREND"]:
            if val < peak_drop_trigger:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"
                trough_val = val
            else:
                current_state = "🟢 STRONG BULLISH TREND"

        elif current_state == "🔴 STRONG BEARISH TREND (Rally Stopped)":
            if val > trough_rise_trigger:
                current_state = "🟢 STRONG BULLISH TREND"
                peak_val = val
            else:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"

        states.append(f"{phase_prefix} {current_state}")

    df["Flip_Status"] = states
    return df


def calculate_dynamic_hints(df_in):
    df = df_in.copy()
    hints = []
    for h_norm, h_diff in zip(df["HAM_Normal"], df["HAM_Diff_Kalman"]):
        if h_diff > 1.0:
            hints.append("🔥 Strong Bullish Expansion")
        elif h_diff < -1.0:
            hints.append("❄️ Bearish Expansion")
        elif abs(h_diff) <= 0.2:
            hints.append("🎯 Equilibrium Zone")
        else:
            hints.append("⚖️ Neutral Momentum")

    df["HAM_Hint"] = hints
    return df


# =====================================================================
# DUAL-SOURCE DATA FETCH ENGINE (BTC-USD)
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
        "Both primary and fallback endpoints failed to return sufficient candles."
    )


# Fetch Data
try:
    with st.spinner("🔄 Fetching BTC-USD (1H) Data & Computing Engine..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# STRICT 50:50 LEARN vs PREDICT PIPELINE WITH VWAP + HURST HUBBLE
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df = apply_heikin_ashi(df)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

# 1. LEARN PHASE (First 50%)
learn_close = df_learn["Close"].to_numpy()
df_learn["Hurst_Normal"] = calculate_rolling_hurst_vectorized(learn_close, window=30)
kalman_base_learn, last_x_base, last_p_base = apply_kalman_filter_custom(
    learn_close, initial_p=50.0, q_val=0.0005, r_val=0.2
)
mom_learn, last_x_mom, last_p_mom = apply_kalman_filter_custom(
    learn_close - kalman_base_learn, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_learn["HAM_Normal"] = mom_learn * (df_learn["Hurst_Normal"].to_numpy() * 2.0)

learn_ha_close = df_learn["HA_Close"].to_numpy()
df_learn["Hurst_HA"] = calculate_rolling_hurst_vectorized(learn_ha_close, window=30)
kalman_ha_learn, last_x_ha, last_p_ha = apply_kalman_filter_custom(
    learn_ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
)
mom_ha_learn, last_x_ha_mom, last_p_ha_mom = apply_kalman_filter_custom(
    learn_ha_close - kalman_ha_learn, initial_p=0.50, q_val=0.001, r_val=0.1
)
df_learn["HAM_HeikinAshi"] = mom_ha_learn * (df_learn["Hurst_HA"].to_numpy() * 2.0)

raw_diff_learn = df_learn["HAM_Normal"] - df_learn["HAM_HeikinAshi"]
df_learn["HAM_Diff_Raw"] = raw_diff_learn
kalman_diff_learn, last_x_diff, last_p_diff = apply_kalman_filter_custom(
    raw_diff_learn.to_numpy(), initial_p=0.50, q_val=0.0001, r_val=0.1
)
df_learn["HAM_Diff_Kalman"] = kalman_diff_learn


# 2. PREDICT PHASE (Last 50%)
predict_close = df_predict["Close"].to_numpy()
df_predict["Hurst_Normal"] = calculate_rolling_hurst_vectorized(predict_close, window=30)
kalman_base_pred, _, _ = apply_kalman_filter_custom(
    predict_close, initial_p=50.0, q_val=0.0005, r_val=0.2, last_x=last_x_base, last_p=last_p_base
)
mom_pred, _, _ = apply_kalman_filter_custom(
    predict_close - kalman_base_pred, initial_p=0.50, q_val=0.001, r_val=0.1, last_x=last_x_mom, last_p=last_p_mom
)
df_predict["HAM_Normal"] = mom_pred * (df_predict["Hurst_Normal"].to_numpy() * 2.0)

predict_ha_close = df_predict["HA_Close"].to_numpy()
df_predict["Hurst_HA"] = calculate_rolling_hurst_vectorized(predict_ha_close, window=30)
kalman_ha_pred, _, _ = apply_kalman_filter_custom(
    predict_ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2, last_x=last_x_ha, last_p=last_p_ha
)
mom_ha_pred, _, _ = apply_kalman_filter_custom(
    predict_ha_close - kalman_ha_pred, initial_p=0.50, q_val=0.001, r_val=0.1, last_x=last_x_ha_mom, last_p=last_p_ha_mom
)
df_predict["HAM_HeikinAshi"] = mom_ha_pred * (df_predict["Hurst_HA"].to_numpy() * 2.0)

raw_diff_pred = df_predict["HAM_Normal"] - df_predict["HAM_HeikinAshi"]
df_predict["HAM_Diff_Raw"] = raw_diff_pred
kalman_diff_pred, _, _ = apply_kalman_filter_custom(
    raw_diff_pred.to_numpy(), initial_p=0.50, q_val=0.0001, r_val=0.1, last_x=last_x_diff, last_p=last_p_diff
)
df_predict["HAM_Diff_Kalman"] = kalman_diff_pred

df = pd.concat([df_learn, df_predict])

# ---------------------------------------------------------------------
# 🌌 HYBRID VWAP DISPERSION + HURST ADAPTIVE HUBBLE EXPANSION
# ---------------------------------------------------------------------
H0_const = 70.0
Lambda_const = 1.1056e-52
c_speed = 299792458.0
G_const = 6.6743e-11

# 1. Rolling VWAP
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# 2. Scale Factor a(t) using VWAP Dispersion & Hurst Multiplier
vwap_dispersion = np.abs((df["Close"] - df["VWAP"]) / df["VWAP"]) * 100.0
df["HAM_Expansion_a"] = 1.0 + (vwap_dispersion * (df["Hurst_Normal"].to_numpy() * 2.0))

# 3. Hubble Velocity & Gaussian Filter
raw_hubble_vel = H0_const * df["HAM_Expansion_a"].to_numpy()
df["HAM_Hubble_Vel_v"] = apply_gaussian_smoothing(raw_hubble_vel, sigma=gaussian_sigma)

# 4. Cosmic Acceleration
dark_energy_factor = (Lambda_const * (c_speed**2)) / 3.0
matter_gravity_factor = (4.0 * np.pi * G_const) / 3.0
df["HAM_Cosmic_Accel_a_dotdot"] = (dark_energy_factor - matter_gravity_factor) * df["HAM_Expansion_a"]

df = apply_hysteresis_state_machine(df, split_idx=split_idx, reversal_threshold_pct=0.20)
df = calculate_dynamic_hints(df)

# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
df_predict_out = df.iloc[split_idx:].copy()

clean_cols = [
    "Close",
    "HA_Close",
    "VWAP",
    "Hurst_Normal",
    "HAM_Normal",
    "HAM_Expansion_a",
    "HAM_Hubble_Vel_v",
    "HAM_Cosmic_Accel_a_dotdot",
    "HAM_HeikinAshi",
    "HAM_Hint",
    "HAM_Diff_Kalman",
    "HAM_Velocity",
    "HAM_Acceleration",
    "Flip_Status",
]

display_df = pd.DataFrame(index=df_predict_out.index)

for col in clean_cols:
    if col not in ["Flip_Status", "HAM_Hint"]:
        display_df[col] = np.asarray(df_predict_out[col], dtype=float).flatten()
    else:
        display_df[col] = df_predict_out[col]

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

# Metrics Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("BTC Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("BTC VWAP Level", f"${latest_candle['VWAP']:,.2f}")
col3.metric("🌌 Scale Factor (a)", f"{latest_candle['HAM_Expansion_a']:.4f}")
col4.metric(
    f"🔭 Hubble Vel (σ={gaussian_sigma})",
    f"{latest_candle['HAM_Hubble_Vel_v']:.2f} km/s",
)
col5.metric(
    "🚀 Cosmic Accel (ä)", f"{latest_candle['HAM_Cosmic_Accel_a_dotdot']:.4e}"
)

st.divider()

# Interactive Data Frame
st.subheader(f"📋 Dynamic Kinematic Matrix ({len(display_df):,} Out-Of-Sample Candles)")
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "BTC Close ($)", format="$%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close ($)", format="$%.2f"
        ),
        "VWAP": st.column_config.NumberColumn(
            "VWAP Level ($)", format="$%.2f"
        ),
        "Hurst_Normal": st.column_config.NumberColumn("Hurst", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn(
            "Base HAM Normal (Kalman)", format="%.2f"
        ),
        "HAM_Expansion_a": st.column_config.NumberColumn(
            "🌌 Scale Factor a(t)", format="%.4f"
        ),
        "HAM_Hubble_Vel_v": st.column_config.NumberColumn(
            f"🔭 Hubble Vel (Gaussian σ={gaussian_sigma})", format="%.2f"
        ),
        "HAM_Cosmic_Accel_a_dotdot": st.column_config.NumberColumn(
            "🚀 Cosmic Accel (ä)", format="%.4e"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA Signal (Kalman)", format="%.2f"
        ),
        "HAM_Hint": st.column_config.TextColumn("💡 HAM Hint Dynamic"),
        "HAM_Diff_Kalman": st.column_config.NumberColumn(
            "📊 HAM Diff (Kalman)", format="%.2f"
        ),
        "HAM_Velocity": st.column_config.NumberColumn(
            "⚡ Velocity (Kalman Q=1e-6)", format="%.4f"
        ),
        "HAM_Acceleration": st.column_config.NumberColumn(
            "🚀 Acceleration (Δ2)", format="%.4f"
        ),
        "Flip_Status": st.column_config.TextColumn("🎯 Hysteresis Lock"),
    },
    use_container_width=True,
    height=600,
)
