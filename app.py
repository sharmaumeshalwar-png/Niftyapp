import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Nifty 50 Dual 5M HAM (W-50 & W-30) Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚀 NIFTY 50 Dual 5M HAM (Window 50 & Window 30) + Speed Engine")
st.caption(
    "100% Leak-Free Grid with 50-Window HAM, Hurst-Scaled 30-Window HAM & Physics Speed Radar"
)


# =====================================================================
# MATHEMATICAL ENGINES (HA, KALMAN & HURST)
# =====================================================================
def compute_heikin_ashi(df_in: pd.DataFrame) -> pd.DataFrame:
    df_ha = df_in.copy()

    op = df_ha["Open"].to_numpy().flatten()
    hi = df_ha["High"].to_numpy().flatten()
    lo = df_ha["Low"].to_numpy().flatten()
    cl = df_ha["Close"].to_numpy().flatten()

    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0

    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))

    df_ha["HA_Open"] = ha_open
    df_ha["HA_High"] = ha_high
    df_ha["HA_Low"] = ha_low
    df_ha["HA_Close"] = ha_close

    return df_ha


def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x, p = data_array[0], initial_p
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values


def calculate_rolling_hurst_leak_free(price_series, window=30):
    hurst_values = np.full(len(price_series), 0.5)
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()

    for i in range(window, len(price_series)):
        window_data = log_returns[i - window + 1 : i + 1]
        mean_val = np.mean(window_data)
        cum_dev = np.cumsum(window_data - mean_val)
        r_val = np.max(cum_dev) - np.min(cum_dev)
        s_val = np.std(window_data)

        if s_val > 1e-8 and r_val > 0:
            rs_ratio = r_val / s_val
            h = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h, 0.0, 1.0)

    return hurst_values


def compute_ha_ham_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # Physics Velocity & Acceleration
    window = 5
    df_ha["Velocity_Speed"] = df_ha["Close"].diff(window) / window
    df_ha["Acceleration_GForce"] = df_ha["Velocity_Speed"].diff(window) / window

    # Hurst Exponent on 30 Window
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(ha_close, window=30)

    # 1. HAM Baseline (Window 30 with Hurst Scaling)
    kalman_30 = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30 = apply_kalman_filter_custom(
        ha_close - kalman_30, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_30"] = np.array(mom_30) * (df_ha["Hurst_30"].to_numpy() * 2.0)

    # 2. HAM (Window 50 on 5M Close / HA_Close)
    kalman_50 = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0003, r_val=0.3
    )
    mom_50 = apply_kalman_filter_custom(
        ha_close - kalman_50, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_50"] = np.array(mom_50)

    return df_ha


# =====================================================================
# DATA INGESTION
# =====================================================================
@st.cache_data(ttl=60)
def load_market_data():
    df_1h_raw = yf.download(
        tickers="^NSEI", period="60d", interval="1h", progress=False
    )
    df_15m_raw = yf.download(
        tickers="^NSEI", period="60d", interval="15m", progress=False
    )
    df_5m_raw = yf.download(
        tickers="^NSEI", period="60d", interval="5m", progress=False
    )

    processed_dfs = []
    for d in [df_1h_raw, df_15m_raw, df_5m_raw]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

        d = d.dropna(subset=["Open", "High", "Low", "Close"])

        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            d.index = d.index.tz_convert("Asia/Kolkata")

        processed_dfs.append(d)

    return processed_dfs[0], processed_dfs[1], processed_dfs[2]


with st.spinner("Loading Nifty Data & Calculating Dual 5M HAM (W-50 & W-30)..."):
    try:
        df_1h_raw, df_15m_raw, df_5m_raw = load_market_data()
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Features Compute
df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)
df_5m = compute_ha_ham_features(df_5m_raw)

# =====================================================================
# MULTI-TIMEFRAME ALIGNMENT & SIGNAL ENGINE
# =====================================================================
df_15m_grid = df_15m.copy()

# Shift 1H & 5M values by 1 bar for leak-free accuracy
df_15m_grid["1H_HA_Close_Frozen"] = (
    df_1h["HA_Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["1H_HAM"] = (
    df_1h["HA_HAM_30"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["1H_HAM_Prev"] = (
    df_1h["HA_HAM_30"].shift(2).reindex(df_15m_grid.index, method="ffill")
)

df_15m_grid["5M_Close"] = (
    df_5m["Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Hurst_30"] = (
    df_5m["Hurst_30"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_HAM_30"] = (
    df_5m["HA_HAM_30"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_HAM_50"] = (
    df_5m["HA_HAM_50"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Velocity"] = (
    df_5m["Velocity_Speed"].shift(1).reindex(df_15m_grid.index, method="ffill")
)

n = len(df_15m_grid)
h1_curr_arr = df_15m_grid["1H_HAM"].to_numpy()
h1_prev_arr = df_15m_grid["1H_HAM_Prev"].to_numpy()
m15_curr_arr = df_15m_grid["HA_HAM_30"].to_numpy()

ha_close_vals = df_15m_grid["HA_Close"].to_numpy()
ha_open_vals = df_15m_grid["HA_Open"].to_numpy()

hurst_5m_vals = df_15m_grid["5M_Hurst_30"].to_numpy()
ham_5m_30_vals = df_15m_grid["5M_HAM_30"].to_numpy()
ham_5m_50_vals = df_15m_grid["5M_HAM_50"].to_numpy()

signals = ["⚪ NEUTRAL"] * n
regime = ["⚡ ORDERED TREND"] * n
ham_alignment = ["⚪ MIXED"] * n

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    h_5m = hurst_5m_vals[i]
    h5_30 = ham_5m_30_vals[i]
    h5_50 = ham_5m_50_vals[i]

    is_ha_red = ha_close_vals[i] < ha_open_vals[i]

    # Dual 5M HAM Alignment Check (Window 50 & Window 30)
    if h5_50 > 0 and h5_30 > 0:
        ham_alignment[i] = "🟢 DUAL BULLISH (50 & 30 HAM > 0)"
    elif h5_50 < 0 and h5_30 < 0:
        ham_alignment[i] = "🔴 DUAL BEARISH (50 & 30 HAM < 0)"
    elif h5_50 > 0 and h5_30 < 0:
        ham_alignment[i] = "⚡ STRONG RECOVERY (50 HAM Green / 30 Red)"
    else:
        ham_alignment[i] = "⚠️ MOMENTUM WEAKENING (50 HAM Red / 30 Green)"

    # Hurst Filter (Anti-Chop Rule: Hurst < 0.48 -> No Trade)
    if h_5m < 0.48:
        regime[i] = "🌀 CHOPPY / MEAN-REVERTING"
        signals[i] = "🛑 NO TRADE (Low Hurst < 0.48)"
        continue

    regime[i] = "⚡ ORDERED / TRENDING"

    # Kinematic Trend Signals
    if h1_curr > 0 and h1_curr < h1_prev:
        signals[i] = (
            "🔴 REAL TOP (1H Drop)"
            if (m15_curr < 0 or is_ha_red)
            else "🟢 DIP BUY PASS"
        )
    elif h1_curr < 0 and h1_curr > h1_prev:
        signals[i] = (
            "🟢 REAL BOTTOM (1H Rise)"
            if (m15_curr > 0 and not is_ha_red)
            else "🔴 FAKE RALLY PASS"
        )
    elif h1_curr > h1_prev and h1_curr > 0 and h5_50 > 0:
        signals[i] = "🟢 ACCELERATED RALLY 🚀"
    elif h1_curr < h1_prev and h1_curr < 0 and h5_50 < 0:
        signals[i] = "🔴 ACCELERATED DROP 🔥"

df_15m_grid["Market_Regime"] = regime
df_15m_grid["5M_HAM_Alignment"] = ham_alignment
df_15m_grid["Kinematic_Signal"] = signals

df_15m_grid.dropna(inplace=True)
latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime("%Y-%m-%d %H:%M IST")

# =====================================================================
# DISPLAY RADAR & TIMELINE
# =====================================================================
st.markdown("---")
st.subheader("📊 Live Dual 5M HAM Momentum Dashboard")

m_c1, m_c2, m_c3, m_c4 = st.columns(4)

with m_c1:
    st.metric(
        label="5M HAM (Window 50)",
        value=f"{latest['5M_HAM_50']:+.2f}",
        delta="Positive" if latest["5M_HAM_50"] > 0 else "Negative",
    )

with m_c2:
    st.metric(
        label="5M HAM (Scaled Window 30)",
        value=f"{latest['5M_HAM_30']:+.2f}",
        delta="Positive" if latest["5M_HAM_30"] > 0 else "Negative",
    )

with m_c3:
    st.metric(
        label="5M Hurst Exponent (W-30)",
        value=f"{latest['5M_Hurst_30']:.2f}",
        delta="Trending" if latest["5M_Hurst_30"] >= 0.50 else "Choppy",
    )

with m_c4:
    st.metric(
        label="5M Speed (Velocity)",
        value=f"{latest['5M_Velocity']:+.2f} Pts",
    )

st.markdown("---")

col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest["Kinematic_Signal"]
    reg = latest["Market_Regime"]

    if reg == "🌀 CHOPPY / MEAN-REVERTING":
        st.warning(
            f"### Market State: {reg}\n# 🛑 AVOID TRADING\n*(Hurst {latest['5M_Hurst_30']:.2f} < 0.48)*"
        )
    elif "RALLY" in sig or "BOTTOM" in sig:
        st.success(f"### Signal ({latest_time})\n# {sig}\n`{latest['5M_HAM_Alignment']}`")
    elif "DROP" in sig or "TOP" in sig:
        st.error(f"### Signal ({latest_time})\n# {sig}\n`{latest['5M_HAM_Alignment']}`")
    else:
        st.info(f"### Signal ({latest_time})\n# {sig}\n`{latest['5M_HAM_Alignment']}`")

with col_s2:
    summary_table = pd.DataFrame(
        {
            "Metric": [
                "15M HA-Close",
                "5M HAM (Window 50 - Trend)",
                "5M HAM (Window 30 - Baseline)",
                "5M Hurst Exponent (Window 30)",
                "5M Momentum Alignment State",
            ],
            "Live Value": [
                f"{latest['HA_Close']:,.2f}",
                f"{latest['5M_HAM_50']:+.2f}",
                f"{latest['5M_HAM_30']:+.2f}",
                f"{latest['5M_Hurst_30']:.2f}",
                f"{latest['5M_HAM_Alignment']}",
            ],
        }
    )
    st.table(summary_table)

st.markdown("---")
st.subheader("📋 Timeline with Dual 5M HAM Grid (W-50 & W-30)")

clean_cols = [
    "Market_Regime",
    "5M_HAM_Alignment",
    "5M_HAM_50",
    "5M_HAM_30",
    "5M_Hurst_30",
    "HA_Close",
    "Kinematic_Signal",
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(
    columns={
        "Market_Regime": "Regime",
        "5M_HAM_Alignment": "5M HAM State",
        "5M_HAM_50": "5M HAM (W-50)",
        "5M_HAM_30": "5M HAM (W-30)",
        "5M_Hurst_30": "5M Hurst",
        "HA_Close": "15M HA-Close",
        "Kinematic_Signal": "Signal",
    },
    inplace=True,
)

for c in ["5M HAM (W-50)", "5M HAM (W-30)", "5M Hurst", "15M HA-Close"]:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(display_df, use_container_width=True, height=600)
