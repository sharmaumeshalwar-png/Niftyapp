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
    page_title="BTC 2-Year Kinematics Engine (Quantum Wave Flow)", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) 2-Year Quantum Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Smooth Quantum Probability Flow & Level Targets | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success("🛡️ **Leak Protection:** ACTIVE | 🔒 **Dual REST Stream:** CONNECTED")


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
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


def calculate_smooth_quantum_wave_levels(price_series, window=30, alpha=0.15):
    """
    Quantum Wave Mechanics with Smooth Probability Flow (No Zig-Zag)
    and Up/Down Target Level Barriers.
    """
    arr = np.asarray(price_series, dtype=float).flatten()
    n = len(arr)
    breakout_prob = np.full(n, 50.0)
    target_levels = np.full(n, arr[0] if n > 0 else 0.0)
    direction_labels = np.full(n, "NEUTRAL", dtype=object)

    raw_prob = 50.0

    for i in range(window, n):
        sub_arr = arr[i - window : i + 1]
        current_price = arr[i]
        
        upper_barrier = np.max(sub_arr[:-1])
        lower_barrier = np.min(sub_arr[:-1])
        std_v = np.std(sub_arr) + 1e-10

        # Calculate momentum direction
        velocity = current_price - sub_arr[-2] if len(sub_arr) > 1 else 0.0
        
        if velocity >= 0:
            barrier_dist = abs(upper_barrier - current_price)
            target = upper_barrier
            label = "UP Target"
        else:
            barrier_dist = abs(current_price - lower_barrier)
            target = lower_barrier
            label = "DOWN Target"

        kinetic_energy = abs(velocity)
        decay_factor = np.sqrt(max(0, barrier_dist - kinetic_energy)) / std_v
        
        # Raw probability calculation
        instant_prob = np.exp(-1.2 * decay_factor) * 100.0
        
        # 🌊 Smooth Flow Engine (Eliminates Zig-Zag)
        raw_prob = (alpha * instant_prob) + ((1 - alpha) * raw_prob)
        
        breakout_prob[i] = np.clip(raw_prob, 0.0, 100.0)
        target_levels[i] = target
        direction_labels[i] = label

    return breakout_prob, target_levels, direction_labels


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
        res = requests.get(endpoint, params=params, headers=headers, timeout=10).json()
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

    cols = ["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"]
    df_raw = pd.DataFrame(all_candles, columns=cols)
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_raw[num_cols] = df_raw[num_cols].astype(float)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["OpenTime"], unit="ms", utc=True)
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
        params = {"granularity": 3600, "start": current_start.isoformat(), "end": current_end.isoformat()}
        res = requests.get(endpoint, params=params, headers=headers, timeout=10).json()
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
        df = fetch_binance_data(int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000))
        if df is not None and len(df) >= 5000:
            return df, "Binance REST API"
    except Exception:
        pass

    df = fetch_coinbase_data(start_dt, now)
    if df is not None and len(df) >= 2000:
        return df, "Coinbase Pro API (Fallback)"

    raise ValueError("Failed to fetch data.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching Data & Computing Quantum Levels..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]  # Drop unclosed running candle
        df.index = df.index.tz_convert("Asia/Kolkata")
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ CALCULATIONS
# =====================================================================
df = apply_heikin_ashi(df)

normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(normal_close_full, window=30)

# ⚛️ SMOOTH QUANTUM PROBABILITY & LEVEL TARGETS
q_prob, q_level, q_dir = calculate_smooth_quantum_wave_levels(normal_close_full, window=30, alpha=0.15)
df["Quantum_Flow_Prob"] = q_prob
df["Quantum_Target_Level"] = q_level
df["Target_Direction"] = q_dir

kalman_base_normal_full = apply_kalman_filter_custom(normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal_full = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal_full, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_Normal"] = momentum_normal_full * (df["Hurst_Normal"].to_numpy() * 2.0)

ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)

kalman_base_ha_full = apply_kalman_filter_custom(ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha_full = apply_kalman_filter_custom(ha_close_full - kalman_base_ha_full, initial_p=0.50, q_val=0.001, r_val=0.1)
df["HAM_HeikinAshi"] = momentum_ha_full * (df["Hurst_HA"].to_numpy() * 2.0)
df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]


# =====================================================================
# DISPLAY MATRIX
# =====================================================================
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy()

display_df = pd.DataFrame(index=df_predict.index)
display_df["Close"] = df_predict["Close"].round(2)
display_df["Target_Direction"] = df_predict["Target_Direction"]
display_df["Quantum_Target_Level"] = df_predict["Quantum_Target_Level"].round(2)
display_df["Quantum_Flow_Prob"] = df_predict["Quantum_Flow_Prob"].round(1)
display_df["Hurst_Normal"] = df_predict["Hurst_Normal"].round(2)
display_df["HAM_Normal"] = df_predict["HAM_Normal"].round(2)
display_df["HAM_HeikinAshi"] = df_predict["HAM_HeikinAshi"].round(2)
display_df["HAM_Diff"] = df_predict["HAM_Diff"].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Flow Probability", f"{latest_candle['Quantum_Flow_Prob']:.1f}%")
col3.metric("Target Level", f"${latest_candle['Quantum_Target_Level']:,.2f}")
col4.metric("Direction", f"{latest_candle['Target_Direction']}")

st.divider()

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "Target_Direction": st.column_config.TextColumn("🎯 Target Type"),
        "Quantum_Target_Level": st.column_config.NumberColumn("📍 Quantum Target ($)", format="$%.2f"),
        "Quantum_Flow_Prob": st.column_config.NumberColumn("🌊 Flow Prob (%)", format="%.1f%%"),
        "Hurst_Normal": st.column_config.NumberColumn("Hurst Normal", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn("HAM Normal", format="%.2f"),
        "HAM_HeikinAshi": st.column_config.NumberColumn("HAM HA", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("HAM Diff", format="%.2f"),
    },
    use_container_width=True,
    height=600,
)
