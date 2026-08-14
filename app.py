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
    page_title="BTC Kinematics Engine (Anti-Whipsaw)", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) Kinematic Engine (Anti-Whipsaw Regimes)")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Continuous Kinematics | Smoothed Velocity (Δ1) & Acceleration (Δ2) | Multi-Candle Regime Lock"
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
# MATHEMATICAL ENGINES
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

    windows = np.lib.stride_tricks.sliding_window_view(log_returns, window_shape=window)
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


# NEW STOIC ANTI-WHIPSAW FLIP ENGINE
def compute_anti_whipsaw_regime(df_in, smooth_span=3, confirmation_candles=3):
    df = df_in.copy()

    # 1. Raw Derivatives
    raw_vel = df["HAM_Diff"].diff()
    raw_acc = raw_vel.diff()

    # 2. Smooth Derivatives via Exponential Moving Average (Kills Micro-Jitter)
    df["HAM_Velocity"] = raw_vel.ewm(span=smooth_span, adjust=False).mean()
    df["HAM_Acceleration"] = raw_acc.ewm(span=smooth_span, adjust=False).mean()

    # 3. Dynamic Volatility Deadband (ATR Equivalent)
    volatility_buffer = df["HAM_Diff"].abs().rolling(20).mean() * 0.15

    # 4. Persistence Signals (Check last N candles)
    bull_persistent = (df["HAM_Velocity"] > 0) & (df["HAM_Acceleration"] > 0)
    bear_persistent = (df["HAM_Velocity"] < 0) & (df["HAM_Acceleration"] < 0)

    # Rolling confirmation check
    bull_confirmed = bull_persistent.rolling(confirmation_candles).sum() == confirmation_candles
    bear_confirmed = bear_persistent.rolling(confirmation_candles).sum() == confirmation_candles

    conditions = [
        # Real Confirmed Trend Flips (Requires multi-candle alignment & crossing buffer)
        bull_confirmed & (df["HAM_Diff"] > volatility_buffer),
        bear_confirmed & (df["HAM_Diff"] < -volatility_buffer),
        
        # Fakeout / Divergence States
        (df["HAM_Velocity"] < 0) & (df["HAM_Acceleration"] > 0),
        (df["HAM_Velocity"] > 0) & (df["HAM_Acceleration"] < 0),
    ]

    choices = [
        "🟢 STRONG BULLISH REGIME",
        "🔴 STRONG BEARISH REGIME",
        "⚠️ FAKEOUT (Wapas Badhega)",
        "⚠️ FAKEOUT (Wapas Girega)",
    ]

    # Default to Neutral / Consolidated State to PREVENT Rapid Flipping
    df["Flip_Status"] = np.select(
        conditions, choices, default="🟡 NEUTRAL / CONSOLIDATION"
    )
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

    raise ValueError("Both primary and fallback endpoints failed to return sufficient candles.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching Hourly BTC Data & Applying Anti-Whipsaw Engine..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1] # Drop running candle
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# FULL CONTINUOUS KINEMATICS & SMOOTHING
# =====================================================================
df = apply_heikin_ashi(df)

# Normal Close Path
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(normal_close_full, window=30)

kalman_base_normal = apply_kalman_filter_custom(normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_normal = apply_kalman_filter_custom(normal_close_full - kalman_base_normal, initial_p=0.50, q_val=0.001, r_val=0.1)
df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

# HA Close Path
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)

kalman_base_ha = apply_kalman_filter_custom(ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2)
momentum_ha = apply_kalman_filter_custom(ha_close_full - kalman_base_ha, initial_p=0.50, q_val=0.001, r_val=0.1)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

# HAM Diff
df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]

# Anti-Whipsaw Filter
df = compute_anti_whipsaw_regime(df, smooth_span=3, confirmation_candles=3)


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
    "HAM_Diff",
    "HAM_Velocity",
    "HAM_Acceleration",
    "Flip_Status",
]
display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    if col != "Flip_Status":
        display_df[col] = np.asarray(df_predict[col], dtype=float).flatten().round(2)
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
col4.metric("📊 HAM Diff", f"{latest_candle['HAM_Diff']:.2f}")
col5.metric("🎯 Regime Lock", f"{latest_candle['Flip_Status']}")

st.divider()

st.subheader(f"📋 Anti-Whipsaw Kinematic Matrix ({len(display_df):,} Predict Candles)")

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "HA_Close": st.column_config.NumberColumn("HA Close ($)", format="$%.2f"),
        "Hurst_Normal": st.column_config.NumberColumn("Hurst (Normal)", format="%.2f"),
        "Hurst_HA": st.column_config.NumberColumn("Hurst (HA)", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn("Base HAM Normal", format="%.2f"),
        "HAM_HeikinAshi": st.column_config.NumberColumn("HAM HA Signal", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("📊 HAM Diff", format="%.2f"),
        "HAM_Velocity": st.column_config.NumberColumn("⚡ Velocity (Δ1)", format="%.2f"),
        "HAM_Acceleration": st.column_config.NumberColumn("🚀 Acceleration (Δ2)", format="%.2f"),
        "Flip_Status": st.column_config.TextColumn("🎯 Regime / Signal Status"),
    },
    use_container_width=True,
    height=600,
)
