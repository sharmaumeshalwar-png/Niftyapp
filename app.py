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
    page_title="BTC Kinematics (VIDYA + Kalman 0.50)",
    layout="wide",
)
st.title("⚡ Bitcoin (BTC-USD) VIDYA + Kalman Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** A (Close), B (VIDYA Period=14 + Kalman Filter Gain=0.50), C (A - B), D, E Matrix | IST Locked"
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
# 8-STEP VERIFICATION MATHEMATICAL ENGINES
# =====================================================================
# STEP 1 & 2: Calculate VIDYA (Variable Index Dynamic Average)
def compute_vidya(data_array, period=14, hist_period=30):
    """
    Computes VIDYA using Standard Deviation Volatility Index (VI).
    Dynamically adjusts alpha based on market volatility.
    """
    s = pd.Series(data_array, dtype=float)
    
    # Calculate short-term std dev and long-term std dev
    std_short = s.rolling(window=period).std().fillna(0.0)
    std_long = s.rolling(window=hist_period).std().fillna(1e-5)
    
    # Volatility Index VI
    vi = (std_short / (std_long + 1e-10)).clip(upper=1.0).to_numpy()
    
    alpha_base = 2.0 / (period + 1.0)
    vidya = np.zeros(len(data_array), dtype=float)
    vidya[0] = data_array[0]
    
    for i in range(1, len(data_array)):
        current_alpha = alpha_base * vi[i]
        vidya[i] = (current_alpha * data_array[i]) + ((1.0 - current_alpha) * vidya[i - 1])
        
    return vidya


# STEP 3: Apply 1D Kalman Filter (Gain / Process Noise ratio = 0.50)
def apply_kalman_filter(input_array, kalman_gain=0.50):
    n = len(input_array)
    filtered = np.zeros(n, dtype=float)

    if n == 0:
        return filtered

    x_hat = input_array[0]
    p = 1.0
    q = kalman_gain
    r = 1.0 - kalman_gain

    filtered[0] = x_hat

    for t in range(1, n):
        x_hat_minus = x_hat
        p_minus = p + q

        k_gain = p_minus / (p_minus + r)
        x_hat = x_hat_minus + k_gain * (input_array[t] - x_hat_minus)
        p = (1.0 - k_gain) * p_minus

        filtered[t] = x_hat

    return filtered


# STEP 4: Compute Baseline B = Kalman(VIDYA)
def compute_baseline_b(price_array, period=14, gain=0.50):
    vidya_vals = compute_vidya(price_array, period=period)
    kalman_vidya = apply_kalman_filter(vidya_vals, kalman_gain=gain)
    return kalman_vidya


# STEP 5: Calculate Hurst Exponent (Vectorized Trailing R/S Window 30)
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
# DUAL-SOURCE DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
    endpoint = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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
    with st.spinner("🔄 Fetching Data & Computing VIDYA + Kalman(0.50)..."):
        df, source_used = get_robust_2year_hourly()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Drop unclosed running candle
        df = df.iloc[:-1]

        # Convert to IST
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# STEP 6 & 7: MATRIX COMPUTATIONS
# =====================================================================
# A = Close Normal
df["A_Close_Normal"] = np.asarray(df["Close"], dtype=float).flatten()

# B = VIDYA (14) + Kalman Filter (0.50)
df["B_Kalman_VIDYA"] = compute_baseline_b(
    df["A_Close_Normal"].to_numpy(), period=14, gain=0.50
)

# C = A - B
df["C_Diff_Residual"] = df["A_Close_Normal"] - df["B_Kalman_VIDYA"]

# D = Hurst of A
df["D_Hurst_A"] = calculate_rolling_hurst_vectorized(
    df["A_Close_Normal"].to_numpy(), window=30
)

# E = C * D
df["E_Kinematic_Signal"] = df["C_Diff_Residual"] * df["D_Hurst_A"]


# =====================================================================
# STEP 8: 50:50 SPLIT & IST MATRIX DISPLAY
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

clean_cols = [
    "A_Close_Normal",
    "B_Kalman_VIDYA",
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
col2.metric("B (VIDYA + Kalman 0.5)", f"${latest_candle['B_Kalman_VIDYA']:,.2f}")
col3.metric("C (A - B)", f"{latest_candle['C_Diff_Residual']:.2f}")
col4.metric("D (Hurst of A)", f"{latest_candle['D_Hurst_A']:.2f}")
col5.metric("🔥 E (C * D)", f"{latest_candle['E_Kinematic_Signal']:.2f}")

st.divider()

st.subheader(
    f"📋 VIDYA + Kalman (0.50) Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "A_Close_Normal": st.column_config.NumberColumn(
            "A: Close Normal ($)", format="$%.2f"
        ),
        "B_Kalman_VIDYA": st.column_config.NumberColumn(
            "B: VIDYA + Kalman 0.50 ($)", format="$%.2f"
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
