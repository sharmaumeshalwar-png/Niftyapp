from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Sensex Kinematics Engine", layout="wide")
st.title("⚡ BSE Sensex (^BSESN) Kinematics Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Sensex HAM Normal (Kalman Filter Core)"
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
# DATA FETCH ENGINE (SENSEX FROM YAHOO FINANCE)
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_sensex_data():
    """Fetch 1-Hour Interval Sensex (^BSESN) Data."""
    try:
        ticker = yf.Ticker("^BSESN")
        # 1-hour interval data for last 730 days (max allowed by Yahoo for 1h)
        df_raw = ticker.history(period="60d", interval="1h")

        if df_raw.empty:
            return None

        df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Handle timezone to IST
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize("Asia/Kolkata")
        else:
            df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")

        return df_raw
    except Exception:
        return None


# Fetch Data Execution
try:
    with st.spinner("🔄 Fetching BSE Sensex Data from Yahoo Finance..."):
        df = fetch_sensex_data()

        if df is None or len(df) < 50:
            st.error("🚨 Sensex price data fetch failed. Please try again.")
            st.stop()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # Drop incomplete live candle
        df = df.iloc[:-1]

except Exception as e:
    st.error(f"🚨 Engine Processing Error: {e}")
    st.stop()


# =====================================================================
# CALCULATIONS
# =====================================================================
# Base HAM Normal Signal for Sensex
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


# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

clean_cols = ["Close", "HAM_Normal"]

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
col1, col2 = st.columns(2)
col1.metric("Sensex Locked Close Price", f"₹{latest_candle['Close']:,.2f}")
col2.metric("HAM Normal (Kalman)", f"{latest_candle['HAM_Normal']:.4f}")

st.divider()

# Interactive Data Frame
st.subheader(
    f"📋 Sensex Kinematics Matrix ({len(display_df):,} Locked Hourly Candles)"
)
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Sensex Close (₹)", format="₹%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "HAM Normal (Kalman)", format="%.4f"
        ),
    },
    use_container_width=True,
    height=600,
)
