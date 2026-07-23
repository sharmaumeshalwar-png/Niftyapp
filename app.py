import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Nifty HA Triple-Timeframe Kinematic Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ NIFTY 50 Heikin-Ashi Triple Engine + Anti-Chop Radar")
st.caption(
    "100% Leak-Free Grid with 5M Hurst-Price & Supertrend (7,3)"
)


# =====================================================================
# MATHEMATICAL ENGINES & CHOPMETRICS
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


def calculate_efficiency_ratio(price_series, window=14):
    """Efficiency Ratio (Kaufman ER): Net Change / Total Sum of Individual Changes"""
    s = pd.Series(price_series)
    net_change = (s - s.shift(window)).abs()
    sum_individual_changes = (s - s.shift(1)).abs().rolling(window=window).sum()
    er = net_change / (sum_individual_changes + 1e-8)
    return er.fillna(0.0).to_numpy()


def calculate_supertrend_custom(series, period=7, multiplier=3.0):
    """Calculates Supertrend (7, 3) on custom price series"""
    s = pd.Series(series)
    diff = s.diff().abs()
    atr = diff.rolling(window=period).mean().fillna(0.0).to_numpy()
    vals = s.to_numpy()

    n = len(vals)
    upper_band = vals + (multiplier * atr)
    lower_band = vals - (multiplier * atr)

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for Bullish, -1 for Bearish

    for i in range(1, n):
        # Final Upper Band
        if upper_band[i] < final_upper[i - 1] or vals[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band
        if lower_band[i] > final_lower[i - 1] or vals[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction Trailing Logic
        if direction[i - 1] == 1:
            if vals[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if vals[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    return supertrend, direction


def compute_ha_ham_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    df_ha["Hurst"] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    df_ha["ER"] = calculate_efficiency_ratio(ha_close, window=14)

    kalman = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    momentum = apply_kalman_filter_custom(
        ha_close - kalman, initial_p=0.50, q_val=0.001, r_val=0.1
    )

    df_ha["Kalman_Price"] = kalman
    df_ha["HA_HAM"] = np.array(momentum) * (df_ha["Hurst"].to_numpy() * 2.0)
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


with st.spinner("Fetching Extended (60-Day) Nifty 50 Intraday Data..."):
    try:
        df_1h_raw, df_15m_raw, df_5m_raw = load_market_data()
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Compute Heikin-Ashi, HAM & Chop Metrics
df_1h = compute_ha_ham_features(df_1h_raw)
df_15m = compute_ha_ham_features(df_15m_raw)
df_5m = compute_ha_ham_features(df_5m_raw)

# =====================================================================
# MULTI-TIMEFRAME ALIGNMENT ENGINE (100% LEAK-FREE)
# =====================================================================
df_15m_grid = df_15m.copy()

# 1H Data shifted by 1 bar BEFORE forward fill
df_15m_grid["1H_HA_Close_Frozen"] = (
    df_1h["HA_Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["HA_HAM_1H_Frozen"] = (
    df_1h["HA_HAM"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["HA_HAM_1H_Prev"] = (
    df_1h["HA_HAM"].shift(2).reindex(df_15m_grid.index, method="ffill")
)

# 5M Data shifted by 1 bar before reindexing
df_15m_grid["5M_HA_Close"] = (
    df_5m["HA_Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_HA_HAM"] = (
    df_5m["HA_HAM"].shift(1).reindex(df_15m_grid.index, method="ffill")
)

# ---------------------------------------------------------------------
# EXACT USER FORMULA: (5M HA-Close * Hurst) -> Supertrend (7, 3)
# ---------------------------------------------------------------------
df_15m_grid["Composite_Signal"] = (
    df_15m_grid["Hurst"] * df_15m_grid["5M_HA_HAM"]
)

# 5M HA Close * Hurst
df_15m_grid["5M_Hurst_Price"] = (
    df_15m_grid["5M_HA_Close"] * df_15m_grid["Hurst"]
)

# Supertrend (7,3) on (5M HA Close * Hurst)
st_vals, st_dirs = calculate_supertrend_custom(
    df_15m_grid["5M_Hurst_Price"].to_numpy(), period=7, multiplier=3.0
)
df_15m_grid["Supertrend_7_3"] = st_vals
df_15m_grid["Supertrend_Dir"] = st_dirs

# HAM Difference
df_15m_grid["HAM_Diff"] = (
    df_15m_grid["HA_HAM_1H_Frozen"] - df_15m_grid["HA_HAM"]
)

# Delta Momentum
df_15m_grid["HA_Close_Diff_15M"] = (
    df_15m_grid["HA_Close"] - df_15m_grid["HA_Close"].shift(1)
)
df_15m_grid["15M_Delta_Momentum"] = (
    df_15m_grid["HA_Close_Diff_15M"] * df_15m_grid["HA_HAM"]
)

n = len(df_15m_grid)
h1_curr_arr = df_15m_grid["HA_HAM_1H_Frozen"].to_numpy()
h1_prev_arr = df_15m_grid["HA_HAM_1H_Prev"].to_numpy()
m15_curr_arr = df_15m_grid["HA_HAM"].to_numpy()

ha_close_vals = df_15m_grid["HA_Close"].to_numpy()
ha_open_vals = df_15m_grid["HA_Open"].to_numpy()

hurst_vals = df_15m_grid["Hurst"].to_numpy()
er_vals = df_15m_grid["ER"].to_numpy()

signals = ["⚪ NEUTRAL"] * n
regime = ["⚡ TRENDING"] * n

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    h_val = hurst_vals[i]
    e_val = er_vals[i]

    is_ha_red = ha_close_vals[i] < ha_open_vals[i]

    # Chop Detection Rule
    if h_val < 0.48 or e_val < 0.18:
        regime[i] = "⚠️ CHOPPY ZONE"
        signals[i] = "🛑 NO TRADE (SideWays Market)"
        continue

    regime[i] = "⚡ TRENDING"

    if h1_curr > 0 and h1_curr < h1_prev:
        signals[i] = (
            "🔴 REAL TOP (1H Drop + 15M Red)"
            if (m15_curr < 0 or is_ha_red)
            else "🟢 TRAP PASS (15M Bullish / Dip Buy)"
        )
    elif h1_curr < 0 and h1_curr > h1_prev:
        signals[i] = (
            "🟢 REAL BOTTOM (1H Rise + 15M Green)"
            if (m15_curr > 0 and not is_ha_red)
            else "🔴 TRAP PASS (15M Bearish / Fake Rally)"
        )
    elif h1_curr > h1_prev and h1_curr > 0:
        signals[i] = "🟢 ACCELERATED RALLY"
    elif h1_curr < h1_prev and h1_curr < 0:
        signals[i] = "🔴 ACCELERATED DROP"

df_15m_grid["Market_Regime"] = regime
df_15m_grid["Instant_Kinematic_Signal"] = signals
df_15m_grid.dropna(
    subset=[
        "HA_HAM",
        "HA_HAM_1H_Frozen",
        "5M_HA_Close",
        "5M_HA_HAM",
        "Composite_Signal",
        "5M_Hurst_Price",
        "Supertrend_7_3",
    ],
    inplace=True,
)

latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime("%Y-%m-%d %H:%M IST")

# =====================================================================
# DISPLAY MATRIX
# =====================================================================
st.markdown("---")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest["Instant_Kinematic_Signal"]
    reg = latest["Market_Regime"]

    if reg == "⚠️ CHOPPY ZONE":
        st.warning(
            f"### Market Regime: {reg}\n# 🛑 AVOID TRADING\n*(Low Hurst {latest['Hurst']:.2f} / Low ER {latest['ER']:.2f})*"
        )
    elif "REAL BOTTOM" in sig or "TRAP PASS (15M Bullish" in sig or "RALLY" in sig:
        st.success(f"### Live Signal ({latest_time})\n# {sig}")
    elif "REAL TOP" in sig or "TRAP PASS (15M Bearish" in sig or "DROP" in sig:
        st.error(f"### Live Signal ({latest_time})\n# {sig}")
    else:
        st.info(f"### Live Signal ({latest_time})\n# {sig}")

with col_s2:
    summary_data = {
        "Metric": [
            "HA Close Price",
            "HA HAM Momentum",
            "Hurst Exponent (Chop Filter)",
            "Efficiency Ratio (ER)",
            "Composite Signal (Hurst * 5M HAM)",
            "5M Close * Hurst",
            "Supertrend (7, 3)",
        ],
        "1-Hour (Locked)": [
            f"{latest['1H_HA_Close_Frozen']:,.2f}",
            f"{latest['HA_HAM_1H_Frozen']:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ],
        "15-Min (Live)": [
            f"{latest['HA_Close']:,.2f}",
            f"{latest['HA_HAM']:.2f}",
            f"{latest['Hurst']:.2f}",
            f"{latest['ER']:.2f}",
            f"{latest['Composite_Signal']:.2f}",
            f"{latest['5M_Hurst_Price']:,.2f}",
            f"{latest['Supertrend_7_3']:,.2f}",
        ],
        "5-Min (Live)": [
            f"{latest['5M_HA_Close']:,.2f}",
            f"{latest['5M_HA_HAM']:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ],
    }
    st.table(pd.DataFrame(summary_data))

st.markdown("---")

st.subheader(
    "📋 Nifty 50 Timeline with 5M Close * Hurst & Supertrend (7,3)"
)

# Column ordering: 5M HA-Close -> (5M Close * Hurst) -> Supertrend (7,3)
clean_cols = [
    "Market_Regime",
    "Hurst",
    "ER",
    "Composite_Signal",
    "1H_HA_Close_Frozen",
    "HA_Close",
    "5M_HA_Close",
    "5M_Hurst_Price",  # <--- 5M Close ke theek NEXT column
    "Supertrend_7_3",  # <--- Supertrend (7,3)
    "HA_HAM_1H_Frozen",
    "HA_HAM",
    "5M_HA_HAM",
    "Instant_Kinematic_Signal",
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(
    columns={
        "Market_Regime": "Regime",
        "Hurst": "Hurst Exponent",
        "ER": "Eff. Ratio",
        "Composite_Signal": "Composite (Hurst * 5M)",
        "1H_HA_Close_Frozen": "1H HA-Close",
        "HA_Close": "15M HA-Close",
        "5M_HA_Close": "5M HA-Close",
        "5M_Hurst_Price": "5M Close * Hurst",
        "Supertrend_7_3": "Supertrend (7,3)",
        "HA_HAM_1H_Frozen": "1H HAM",
        "HA_HAM": "15M HAM",
        "5M_HA_HAM": "5M HAM",
        "Instant_Kinematic_Signal": "Kinematic Signal",
    },
    inplace=True,
)

for c in [
    "Hurst Exponent",
    "Eff. Ratio",
    "Composite (Hurst * 5M)",
    "1H HA-Close",
    "15M HA-Close",
    "5M HA-Close",
    "5M Close * Hurst",
    "Supertrend (7,3)",
    "1H HAM",
    "15M HAM",
    "5M HAM",
]:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(display_df, use_container_width=True, height=650)
