import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="Nifty 50 Phase-Attractor Kinematics Engine", layout="wide"
)
st.title("⚡ NIFTY 50 Engine (Phase-Attractor Base + Q=0.0001 Kalman)")
st.write(
    "🎯 **1-Hour Timeframe Engine:** Phase Attractor Value as Primary Base → "
    "Secondary Kalman (Q=0.0001) → Spread Difference | 50:50 Learn:Predict Split | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Kalman Setting:** "
    "Q = 0.0001 across all tiers"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.0001, r_val=0.1
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


def calculate_phase_space_attractor_value(ham_series, tau=1):
    """
    Phase-Space Attractor Reconstruction Engine.
    Reconstructs 3D Vector Space [HAM_t, HAM_{t-tau}, HAM_{t-2tau}] and calculates
    Pure Energy Drift Values.
    """
    ham = np.asarray(ham_series, dtype=float).flatten()
    n = len(ham)
    attractor_value = np.zeros(n)

    for i in range(2 * tau, n):
        # Phase Space Vector: [x, y, z]
        x = ham[i]
        y = ham[i - tau]
        z = ham[i - 2 * tau]

        # Velocity Vectors
        v1 = x - y
        v2 = y - z

        # Attractor Magnitude Radius
        attractor_radius = np.sqrt(x**2 + y**2 + z**2) + 1e-10

        # Phase Space Energy Drift Value Calculation
        energy_drift = (x * v1 + y * v2) / attractor_radius
        attractor_value[i] = energy_drift

    return attractor_value


# =====================================================================
# NIFTY 50 DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_nifty_data():
    ticker = "^NSEI"
    df_raw = yf.download(
        tickers=ticker, period="2y", interval="1h", progress=False
    )

    if df_raw.empty:
        raise ValueError("YFinance API se Nifty ka data nahi mila.")

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

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
        df = df.iloc[:-1]  # Drop active unclosed candle
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ FULL KINEMATICS & PHASE ATTRACTOR PIPELINE (Q = 0.0001)
# =====================================================================
df = apply_heikin_ashi(df)

# --- NORMAL HAM ---
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)

kalman_base_normal = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0001, r_val=0.2
)
momentum_normal = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal,
    initial_p=0.50,
    q_val=0.0001,
    r_val=0.1,
)

df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

# --- HEIKIN ASHI HAM ---
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close_full, window=30)

kalman_base_ha = apply_kalman_filter_custom(
    ha_close_full, initial_p=50.0, q_val=0.0001, r_val=0.2
)
momentum_ha = apply_kalman_filter_custom(
    ha_close_full - kalman_base_ha, initial_p=0.50, q_val=0.0001, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]

# --- 🚀 STEP 1: PHASE ATTRACTOR VALUE (Base Series) ---
df["Phase_Attractor_Value"] = calculate_phase_space_attractor_value(
    df["HAM_Normal"].to_numpy(), tau=1
)

# --- 🚀 STEP 2: KALMAN FILTER ON PHASE ATTRACTOR VALUE (Q = 0.0001) ---
df["Kalman_Phase_Attractor"] = apply_kalman_filter_custom(
    df["Phase_Attractor_Value"].to_numpy(), initial_p=1.0, q_val=0.0001, r_val=0.05
)

# --- 🚀 STEP 3: PHASE ATTRACTOR MINUS KALMAN PHASE ATTRACTOR ---
df["Phase_Attractor_Spread"] = (
    df["Phase_Attractor_Value"] - df["Kalman_Phase_Attractor"]
)


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_predict.dropna(subset=["Hurst_Normal", "Hurst_HA"], inplace=True)

st.success(
    f"🟢 **Synced via Yahoo Finance (^NSEI): {total_candles:,} Hourly Candles** | "
    f"🧠 **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:** {len(df_predict):,}"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
display_df = pd.DataFrame(index=df_predict.index)
display_df["Close"] = df_predict["Close"].round(2)
display_df["HA_Close"] = df_predict["HA_Close"].round(2)
display_df["Hurst_Normal"] = df_predict["Hurst_Normal"].round(2)
display_df["HAM_Normal"] = df_predict["HAM_Normal"].round(2)
display_df["HAM_HeikinAshi"] = df_predict["HAM_HeikinAshi"].round(2)
display_df["HAM_Diff"] = df_predict["HAM_Diff"].round(2)

# New Phase Attractor Pipeline Columns (Rounded to 4 Decimals)
display_df["Phase_Attractor_Value"] = df_predict["Phase_Attractor_Value"].round(4)
display_df["Kalman_Phase_Attractor"] = df_predict["Kalman_Phase_Attractor"].round(4)
display_df["Phase_Attractor_Spread"] = df_predict["Phase_Attractor_Spread"].round(4)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# Metric Cards Display
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nifty Close Price", f"₹{latest_candle['Close']:,.2f}")
col2.metric("Phase Attractor Value", f"{latest_candle['Phase_Attractor_Value']:.4f}")
col3.metric("Kalman (Q=0.0001)", f"{latest_candle['Kalman_Phase_Attractor']:.4f}")
col4.metric("💥 Phase Attractor Spread", f"{latest_candle['Phase_Attractor_Spread']:.4f}")

st.divider()

st.subheader(
    f"📋 50:50 Matrix with Q=0.0001 Kalman Pipeline ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price (₹)", format="₹%.2f"),
        "HA_Close": st.column_config.NumberColumn("HA Close (₹)", format="₹%.2f"),
        "Hurst_Normal": st.column_config.NumberColumn("Hurst (Normal)", format="%.2f"),
        "HAM_Normal": st.column_config.NumberColumn("Base HAM Normal", format="%.2f"),
        "HAM_HeikinAshi": st.column_config.NumberColumn("HAM HA Signal", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("HAM Diff", format="%.2f"),
        "Phase_Attractor_Value": st.column_config.NumberColumn("🌀 Phase Attractor Base", format="%.4f"),
        "Kalman_Phase_Attractor": st.column_config.NumberColumn("🛡️ Kalman (Q=0.0001)", format="%.4f"),
        "Phase_Attractor_Spread": st.column_config.NumberColumn("⚡ Spread (Base - Kalman)", format="%.4f"),
    },
    use_container_width=True,
    height=600,
)
