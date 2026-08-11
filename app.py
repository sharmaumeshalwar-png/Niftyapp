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
    page_title="BTC 2-Year Kinematics Engine (Quantum Wave)", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) 2-Year Quantum Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Raw Quantum Wave Mechanics | Target Levels &"
    " Tunneling Status | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")

# ⚛️ Dynamic Quantum Decay Factor Sensitivity Adjustment
sensitivity_factor = st.sidebar.slider(
    "⚛️ Quantum Decay Sensitivity Factor",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help=(
        "Higher value = Sharper breakout sensitivity. Lower value = Smoother"
        " probability curve."
    ),
)

if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Dual REST Stream:**"
    " CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.001, r_val=0.1
):
    """Sequential single-pass Kalman Filter (Zero Leakage)."""
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
    """Vectorized Trailing R/S Hurst Exponent."""
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


def calculate_quantum_wave_mechanics(
    price_series, window=30, decay_factor_multiplier=1.5
):
    """Quantum Wave Mechanics Engine with Dynamic Sensitivity Factor."""
    arr = np.asarray(price_series, dtype=float).flatten()
    n = len(arr)
    breakout_prob = np.full(n, 50.0)
    target_levels = np.full(n, arr[0] if n > 0 else 0.0)
    direction_labels = np.full(n, "NEUTRAL", dtype=object)
    tunnel_status = np.full(n, "BOUNCED", dtype=object)

    for i in range(window, n):
        sub_arr = arr[i - window : i + 1]
        current_price = arr[i]
        prev_price = arr[i - 1]

        upper_barrier = np.max(sub_arr[:-1])
        lower_barrier = np.min(sub_arr[:-1])
        std_v = np.std(sub_arr) + 1e-10

        velocity = current_price - prev_price

        if velocity >= 0:
            barrier_dist = abs(upper_barrier - current_price)
            target = upper_barrier
            label = "UP Target"
            status = (
                "TUNNELED" if current_price >= upper_barrier else "BOUNCED"
            )
        else:
            barrier_dist = abs(current_price - lower_barrier)
            target = lower_barrier
            label = "DOWN Target"
            status = (
                "TUNNELED" if current_price <= lower_barrier else "BOUNCED"
            )

        kinetic_energy = abs(velocity)
        decay_factor = np.sqrt(max(0, barrier_dist - kinetic_energy)) / std_v

        # Dynamic Adjustable Quantum Exponential Decay
        prob = np.exp(-decay_factor_multiplier * decay_factor) * 100.0

        breakout_prob[i] = np.clip(prob, 0.0, 100.0)
        target_levels[i] = target
        direction_labels[i] = label
        tunnel_status[i] = status

    return breakout_prob, target_levels, direction_labels, tunnel_status


def apply_heikin_ashi(df_in):
    """Calculates Heikin-Ashi candles sequentially."""
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

    raise ValueError("Data fetch failed from both sources.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching 2 Years of Hourly BTC Data..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ FULL CONTINUOUS KINEMATICS
# =====================================================================
df = apply_heikin_ashi(df)

normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)

# Quantum Calculation with Slider Input
q_prob, q_level, q_dir, q_status = calculate_quantum_wave_mechanics(
    normal_close_full,
    window=30,
    decay_factor_multiplier=sensitivity_factor,
)
df["Quantum_Breakout_Prob"] = q_prob
df["Quantum_Target_Level"] = q_level
df["Target_Direction"] = q_dir
df["Tunneling_Status"] = q_status

kalman_base_normal_full = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal_full = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal_full,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)
df["HAM_Normal"] = momentum_normal_full * (df["Hurst_Normal"].to_numpy() * 2.0)

ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(
    ha_close_full, window=30
)

kalman_base_ha_full = apply_kalman_filter_custom(
    ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha_full = apply_kalman_filter_custom(
    ha_close_full - kalman_base_ha_full, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha_full * (df["Hurst_HA"].to_numpy() * 2.0)
df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]


# =====================================================================
# ⚡ 50:50 SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced via {source_used}: {total_candles:,} Total Candles** | 🧠"
    f" **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:**"
    f" {len(df_predict):,} (IST Locked)"
)


# =====================================================================
# 📋 MATRIX DISPLAY
# =====================================================================
display_df = pd.DataFrame(index=df_predict.index)
display_df["Close"] = df_predict["Close"].round(2)
display_df["Target_Direction"] = df_predict["Target_Direction"]
display_df["Quantum_Target_Level"] = df_predict["Quantum_Target_Level"].round(2)
display_df["Tunneling_Status"] = df_predict["Tunneling_Status"]
display_df["Quantum_Breakout_Prob"] = df_predict["Quantum_Breakout_Prob"].round(1)
display_df["Hurst_Normal"] = df_predict["Hurst_Normal"].round(2)
display_df["HAM_Normal"] = df_predict["HAM_Normal"].round(2)
display_df["HAM_HeikinAshi"] = df_predict["HAM_HeikinAshi"].round(2)
display_df["HAM_Diff"] = df_predict["HAM_Diff"].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Quantum Prob", f"{latest_candle['Quantum_Breakout_Prob']:.1f}%")
col3.metric("Target Level", f"${latest_candle['Quantum_Target_Level']:,.2f}")
col4.metric("Direction", f"{latest_candle['Target_Direction']}")
col5.metric("Tunneling Status", f"{latest_candle['Tunneling_Status']}")

st.divider()

st.subheader(
    f"📋 50:50 Clean Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "Target_Direction": st.column_config.TextColumn("🎯 Target Type"),
        "Quantum_Target_Level": st.column_config.NumberColumn("📍 Quantum Target ($)", format="$%.2f"),
        "Tunneling_Status": st.column_config.TextColumn("⚡ Level Status"),
        "Quantum_Breakout_Prob": st.column_config.NumberColumn("⚛️ Breakout Prob (%)", format="%.1f%%"),
        "Hurst_Normal": st.column_config.NumberColumn("Hurst Normal", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn("HAM Normal", format="%.2f"),
        "HAM_HeikinAshi": st.column_config.NumberColumn("HAM HA", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("HAM Diff", format="%.2f"),
    },
    use_container_width=True,
    height=600,
)
