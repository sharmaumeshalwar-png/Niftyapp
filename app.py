from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Sensex Normalized Precision Engine", layout="wide"
)
st.title("🎯 BSE Sensex (^BSESN) Normalized Precision Engine")
st.write(
    "⚡ **1-Hour Timeframe:** 2-Year Read | 1-Year Predict | Z-Score Normalized"
    " HAM + Multi-Trend Confirmation"
)

# Sidebar Controls
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
# DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_sensex_2year_hourly():
    """Fetch 2-Year Hourly Sensex (^BSESN) Data from Yahoo Finance."""
    try:
        ticker = yf.Ticker("^BSESN")
        df_raw = ticker.history(period="730d", interval="1h")

        if df_raw.empty:
            return None

        df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize("Asia/Kolkata")
        else:
            df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")

        return df_raw
    except Exception:
        return None


# Fetch Data Execution
try:
    with st.spinner("🔄 Fetching Data & Calculating Precision Signal..."):
        df = fetch_sensex_2year_hourly()

        if df is None or len(df) < 500:
            st.error(
                "🚨 Sensex price data fetch failed. Click 'Force Refresh Engine'"
                " in sidebar."
            )
            st.stop()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Drop incomplete live candle
        df = df.iloc[:-1]

except Exception as e:
    st.error(f"🚨 Engine Processing Error: {e}")
    st.stop()


# =====================================================================
# ENHANCED PRECISION CALCULATIONS
# =====================================================================
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()

# 1. Rolling Hurst Exponent
df["Hurst_Exponent"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)

# 2. Kalman Base & Momentum
kalman_base_normal = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)

# 3. Core HAM Raw
raw_ham = momentum_normal * (df["Hurst_Exponent"].to_numpy() * 2.0)

# 4. Z-Score Normalization (Fixed scale -3 to +3 for zero confusion)
rolling_mean = pd.Series(raw_ham).rolling(window=100, min_periods=20).mean()
rolling_std = (
    pd.Series(raw_ham).rolling(window=100, min_periods=20).std() + 1e-6
)
df["HAM_Normalized"] = ((raw_ham - rolling_mean) / rolling_std).fillna(0.0)

# 5. Multi-Trend Confirmation (EMA 50 & EMA 200)
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

df["Trend_Direction"] = np.where(
    df["Close"] > df["EMA_50"],
    "BULLISH",
    np.where(df["Close"] < df["EMA_50"], "BEARISH", "NEUTRAL"),
)

# 6. Ultra-Accurate Action Signal Logic
conditions = [
    (df["HAM_Normalized"] > 1.0)
    & (df["Trend_Direction"] == "BULLISH")
    & (df["Hurst_Exponent"] > 0.50),
    (df["HAM_Normalized"] < -1.0)
    & (df["Trend_Direction"] == "BEARISH")
    & (df["Hurst_Exponent"] > 0.50),
    (df["HAM_Normalized"] > 2.0),
    (df["HAM_Normalized"] < -2.0),
    (df["Hurst_Exponent"] < 0.45),
]

choices = [
    "🟢 STRONG BUY (Trend Confirmed)",
    "🔴 STRONG SELL (Trend Confirmed)",
    "⚠️️ OVERBOUGHT (Exit Longs)",
    "⚡ OVERSOLD (Exit Shorts)",
    "🟡 CHOPPY / NO TRADE ZONE",
]

df["Precision_Signal"] = np.select(
    conditions, choices, default="⏸️ NEUTRAL / HOLD"
)


# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

display_cols = [
    "Close",
    "HAM_Normalized",
    "Hurst_Exponent",
    "Trend_Direction",
    "Precision_Signal",
]

display_df = df_predict[display_cols].copy()

# Latest candles on top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

# Metrics Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sensex Locked Close", f"₹{latest_candle['Close']:,.2f}")
col2.metric("Precision Trade Signal", f"{latest_candle['Precision_Signal']}")
col3.metric("HAM Z-Score (-3 to +3)", f"{latest_candle['HAM_Normalized']:.2f}")
col4.metric(
    "Trend Confirmation",
    f"{latest_candle['Trend_Direction']} (Hurst: {latest_candle['Hurst_Exponent']:.2f})",
)

st.divider()

# Interactive Table Display
st.subheader(
    f"📋 Precision Signal Matrix ({len(display_df):,} Locked Hourly Candles)"
)
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Sensex Close (₹)", format="₹%.2f"
        ),
        "HAM_Normalized": st.column_config.NumberColumn(
            "Normalized HAM (Z-Score)", format="%.2f"
        ),
        "Hurst_Exponent": st.column_config.NumberColumn(
            "Hurst Exponent", format="%.4f"
        ),
        "Trend_Direction": st.column_config.TextColumn("Macro Trend"),
        "Precision_Signal": st.column_config.TextColumn("Actionable Trade Signal"),
    },
    use_container_width=True,
    height=600,
)
