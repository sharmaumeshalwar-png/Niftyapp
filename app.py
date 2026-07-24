import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Nifty 50 Dual HAM (W-30 & W-200)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚀 NIFTY 50 Dual 5M HAM (Window 30 & Window 200) Engine")
st.caption("Exact Same Theoretical Math Matrix — Window 30 vs Window 200")


# =====================================================================
# 1. HEIKIN-ASHI CALCULATION
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


# =====================================================================
# 2. CUSTOM KALMAN FILTER
# =====================================================================
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


# =====================================================================
# 3. LEAK-FREE ROLLING HURST EXPONENT
# =====================================================================
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


# =====================================================================
# 4. CORE FEATURES (WINDOW 30 VS WINDOW 200)
# =====================================================================
def compute_ha_ham_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # Physics Velocity & Acceleration
    window = 5
    df_ha["Velocity_Speed"] = df_ha["Close"].diff(window) / window
    df_ha["Acceleration_GForce"] = df_ha["Velocity_Speed"].diff(window) / window

    # 1. HAM Window 30 (Fast Momentum)
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    kalman_30 = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30 = apply_kalman_filter_custom(
        ha_close - kalman_30, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_30"] = np.array(mom_30) * (df_ha["Hurst_30"].to_numpy() * 2.0)

    # 2. HAM Window 200 (Macro Anchor - EXACT SAME THEORY)
    df_ha["Hurst_200"] = calculate_rolling_hurst_leak_free(
        ha_close, window=200
    )
    kalman_200 = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_200 = apply_kalman_filter_custom(
        ha_close - kalman_200, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_200"] = np.array(mom_200) * (
        df_ha["Hurst_200"].to_numpy() * 2.0
    )

    return df_ha


# =====================================================================
# 5. DATA INGESTION
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


with st.spinner("Fetching Nifty Data & Computing Window 30 vs 200 HAM..."):
    try:
        df_1h_raw, df_15m_raw, df_5m_raw = load_market_data()
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Compute Features
df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)
df_5m = compute_ha_ham_features(df_5m_raw)

# =====================================================================
# 6. LEAK-FREE ALIGNMENT & DISPLAY
# =====================================================================
df_15m_grid = df_15m.copy()

df_15m_grid["5M_HAM_30"] = (
    df_5m["HA_HAM_30"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_HAM_200"] = (
    df_5m["HA_HAM_200"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Hurst_30"] = (
    df_5m["Hurst_30"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Hurst_200"] = (
    df_5m["Hurst_200"].shift(1).reindex(df_15m_grid.index, method="ffill")
)

df_15m_grid.dropna(inplace=True)
latest = df_15m_grid.iloc[-1]

st.markdown("---")
st.subheader("📊 Live Numeric HAM Values (Window 30 vs Window 200)")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="5M HAM (Window 30)",
        value=f"{latest['5M_HAM_30']:+.2f}",
        delta="Fast Momentum",
    )
with c2:
    st.metric(
        label="5M HAM (Window 200)",
        value=f"{latest['5M_HAM_200']:+.2f}",
        delta="Macro Baseline",
    )
with c3:
    st.metric(
        label="Hurst Exponents (W-30 / W-200)",
        value=f"{latest['5M_Hurst_30']:.2f} / {latest['5M_Hurst_200']:.2f}",
    )

st.markdown("---")
st.subheader("📋 Historical Multi-Timeframe Data Table")

display_cols = [
    "HA_Close",
    "5M_HAM_30",
    "5M_HAM_200",
    "5M_Hurst_30",
    "5M_Hurst_200",
]
table_df = df_15m_grid[display_cols].copy().iloc[::-1]
table_df.columns = [
    "15M HA-Close",
    "5M HAM (W-30)",
    "5M HAM (W-200)",
    "Hurst (W-30)",
    "Hurst (W-200)",
]
table_df.index = table_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(table_df.round(2), use_container_width=True, height=500)
