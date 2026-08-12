import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="Nifty 50 Kinematics Engine (Zero Leakage)", layout="wide"
)
st.title("⚡ NIFTY 50 Pure Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** NIFTY 50 History | Pure HAM Kinematics "
    "Matrix | 50:50 Learn:Predict Split | IST Locked [Strict Zero Leakage & Continuous Warmup]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Data Source:** "
    "Yahoo Finance API (`^NSEI`)"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.001, r_val=0.1
):
    """Sequential single-pass Kalman Filter (No future smoothing / Zero Leakage)."""
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

    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))

    df_out = df_in.copy()
    df_out["HA_Open"] = ha_open
    df_out["HA_High"] = ha_high
    df_out["HA_Low"] = ha_low
    df_out["HA_Close"] = ha_close
    return df_out


# =====================================================================
# NIFTY 50 DATA FETCH ENGINE (YFINANCE)
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_nifty_data():
    """YFinance limit: Hourly data (`1h`) Max 730 days limit tak fetch hoti hai."""
    ticker = "^NSEI"
    df_raw = yf.download(
        tickers=ticker, period="2y", interval="1h", progress=False
    )

    if df_raw.empty:
        raise ValueError("YFinance API se Nifty ka data nahi mil paya.")

    # MultiIndex columns handle karne ke liye (agar YFinance multi-level tuple de)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Timezone conversion to IST
    if df_raw.index.tzinfo is None:
        df_raw.index = df_raw.index.tz_localize("UTC")

    df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")
    return df_raw


# Fetch Data
try:
    with st.spinner("🔄 Fetching Hourly NIFTY 50 Data (`^NSEI`)..."):
        df = fetch_nifty_data()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # 🔒 STRICT LEAKAGE PREVENTION: Drop unclosed running candle
        df = df.iloc[:-1]

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ FULL-LENGTH CONTINUOUS KINEMATICS
# =====================================================================
df = apply_heikin_ashi(df)

# --- PATH A: NORMAL CANDLE KINEMATICS ---
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)

kalman_base_normal_full = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal_full = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal_full,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)

# Base HAM Normal Signal
df["HAM_Normal"] = momentum_normal_full * (df["Hurst_Normal"].to_numpy() * 2.0)

# --- PATH B: HEIKIN-ASHI CANDLE KINEMATICS ---
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)

kalman_base_ha_full = apply_kalman_filter_custom(
    ha_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_ha_full = apply_kalman_filter_custom(
    ha_close_full - kalman_base_ha_full, initial_p=0.50, q_val=0.001, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha_full * (df["Hurst_HA"].to_numpy() * 2.0)

# --- HAM DIFFERENCE (HAM Normal - HAM HA) ---
df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_predict.dropna(subset=["Hurst_Normal", "Hurst_HA"], inplace=True)

st.success(
    f"🟢 **Synced via Yahoo Finance (^NSEI): {total_candles:,} Total Hourly Candles** | 🧠"
    f" **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:**"
    f" {len(df_predict):,} (IST Locked)"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    "Close",
    "HA_Close",
    "Hurst_Normal",
    "Hurst_HA",
    "HAM_Normal",
    "HAM_HeikinAshi",
    "HAM_Diff",
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"₹{latest_candle['Close']:,.2f}")
col2.metric("Base HAM Normal", f"{latest_candle['HAM_Normal']:.2f}")
col3.metric("HAM HA Signal", f"{latest_candle['HAM_HeikinAshi']:.2f}")
col4.metric("📊 HAM Diff (Normal - HA)", f"{latest_candle['HAM_Diff']:.2f}")

st.divider()

st.subheader(
    f"📋 50:50 Clean Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price (₹)", format="₹%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close (₹)", format="₹%.2f"
        ),
        "Hurst_Normal": st.column_config.NumberColumn(
            "Hurst (Normal)", format="%.2f"
        ),
        "Hurst_HA": st.column_config.NumberColumn("Hurst (HA)", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn(
            "Base HAM Normal", format="%.2f"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA Signal", format="%.2f"
        ),
        "HAM_Diff": st.column_config.NumberColumn(
            "📊 HAM Diff (Normal - HA)", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
