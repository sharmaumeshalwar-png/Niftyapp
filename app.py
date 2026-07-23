import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Nifty 1-to-5 Bounded Channel Kinematic Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ NIFTY 50 Bounded Channel Engine (Kalman Q=0.50 on Scaled E)")
st.caption(
    "A: 1-5 Channel | B: Kalman(0.50) | C: Hurst(B) | D: Momentum(A-B) | E: (C * D)"
    " * 1000 | E_Kalman: Kalman(0.50) on E"
)


# =====================================================================
# MATHEMATICAL ENGINES
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


def scale_to_1_to_5_channel(series: pd.Series, window: int = 30) -> pd.Series:
    """Column A: Scale Price into strictly [1.0, 5.0] channel."""
    roll_min = series.rolling(window=window).min()
    roll_max = series.rolling(window=window).max()
    diff = (roll_max - roll_min).replace(0, 1e-8)

    norm = (series - roll_min) / diff
    scaled = 1.0 + (norm * 4.0)
    return scaled.fillna(3.0)


def apply_kalman_filter_q(data_array, q_val=0.50, r_val=0.1, initial_p=1.0):
    """Kalman Filter Implementation."""
    if len(data_array) == 0:
        return np.array([])
    x, p = data_array[0], initial_p
    filtered = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered.append(x)
    return np.array(filtered)


def calculate_rolling_hurst_leak_free(series_in, window=30):
    """Column C: Rolling Hurst Exponent calculated on series B."""
    s = pd.Series(series_in)
    hurst_values = np.full(len(s), 0.5)
    log_returns = (
        np.log((s / s.shift(1)).replace(0, 1e-8)).fillna(0.0).to_numpy()
    )

    for i in range(window, len(s)):
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


def calculate_supertrend_custom(series, period=7, multiplier=3.0):
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
    direction = np.ones(n)

    for i in range(1, n):
        if (
            upper_band[i] < final_upper[i - 1]
            or vals[i - 1] > final_upper[i - 1]
        ):
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if (
            lower_band[i] > final_lower[i - 1]
            or vals[i - 1] < final_lower[i - 1]
        ):
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

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


with st.spinner("Fetching Nifty 50 Data..."):
    try:
        df_1h_raw, df_15m_raw, df_5m_raw = load_market_data()
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# HA Processing
df_1h = compute_heikin_ashi(df_1h_raw)
df_15m = compute_heikin_ashi(df_15m_raw)
df_5m = compute_heikin_ashi(df_5m_raw)

# =====================================================================
# CUSTOM ENGINE IMPLEMENTATION ON 5M TIMEFRAME
# =====================================================================
# A = 5M Close mapped to [1, 5] Channel
df_5m["Col_A_Channel"] = scale_to_1_to_5_channel(df_5m["HA_Close"], window=30)

# B = Kalman Filter on A with Q = 0.50
df_5m["Col_B_Kalman"] = apply_kalman_filter_q(
    df_5m["Col_A_Channel"].to_numpy(), q_val=0.50, r_val=0.1
)

# C = Hurst Exponent on B
df_5m["Col_C_Hurst"] = calculate_rolling_hurst_leak_free(
    df_5m["Col_B_Kalman"].to_numpy(), window=30
)

# D = Weighted Momentum (A - B)
raw_diff = df_5m["Col_A_Channel"] - df_5m["Col_B_Kalman"]
df_5m["Col_D_Weighted_Momentum"] = raw_diff.ewm(span=5, adjust=False).mean()

# E Raw Scaled = (C * D) * 1000
df_5m["Col_E_Raw"] = (
    df_5m["Col_C_Hurst"] * df_5m["Col_D_Weighted_Momentum"]
) * 1000.0

# E Kalman Smoothed = KALMAN FILTER UPDATED TO (Q = 0.50) ON COLUMN E
df_5m["Col_E_Composite"] = apply_kalman_filter_q(
    df_5m["Col_E_Raw"].to_numpy(), q_val=0.50, r_val=0.1
)

# Supertrend on Kalman-Smoothed Column E
st_vals, st_dirs = calculate_supertrend_custom(
    df_5m["Col_E_Composite"].to_numpy(), period=7, multiplier=3.0
)
df_5m["Supertrend_on_E"] = st_vals

# =====================================================================
# MULTI-TIMEFRAME ALIGNMENT (15M MAIN GRID)
# =====================================================================
df_15m_grid = df_15m.copy()

# Forward freeze 5M metrics into 15M grid
for col in [
    "HA_Close",
    "Col_A_Channel",
    "Col_B_Kalman",
    "Col_C_Hurst",
    "Col_D_Weighted_Momentum",
    "Col_E_Composite",
    "Supertrend_on_E",
]:
    df_15m_grid[f"5M_{col}"] = (
        df_5m[col].shift(1).reindex(df_15m_grid.index, method="ffill")
    )

df_15m_grid["1H_HA_Close"] = (
    df_1h["HA_Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)

# Signal Regime Rules
n = len(df_15m_grid)
signals = ["⚪ NEUTRAL"] * n
regime = ["⚡ TRENDING"] * n

comp_vals = df_15m_grid["5M_Col_E_Composite"].to_numpy()
st_e_vals = df_15m_grid["5M_Supertrend_on_E"].to_numpy()
hurst_vals = df_15m_grid["5M_Col_C_Hurst"].to_numpy()

for i in range(2, n):
    e_val = comp_vals[i]
    st_val = st_e_vals[i]
    h_val = hurst_vals[i]

    if h_val < 0.48:
        regime[i] = "⚠️ CHOPPY ZONE"
        signals[i] = "🛑 NO TRADE (Low Hurst / Sideways)"
    elif e_val > st_val:
        regime[i] = "🟢 BULLISH RALLY"
        signals[i] = "🟢 STRONG BUY (Filtered E > Supertrend)"
    elif e_val < st_val:
        regime[i] = "🔴 BEARISH DROP"
        signals[i] = "🔴 STRONG SELL (Filtered E < Supertrend)"

df_15m_grid["Market_Regime"] = regime
df_15m_grid["Kinematic_Signal"] = signals

df_15m_grid.dropna(
    subset=[
        "5M_Col_A_Channel",
        "5M_Col_B_Kalman",
        "5M_Col_C_Hurst",
        "5M_Col_D_Weighted_Momentum",
        "5M_Col_E_Composite",
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
    sig = latest["Kinematic_Signal"]
    reg = latest["Market_Regime"]

    if reg == "⚠️ CHOPPY ZONE":
        st.warning(
            f"### Market Regime: {reg}\n# 🛑 AVOID TRADING\n*(Hurst on B ="
            f" {latest['5M_Col_C_Hurst']:.2f})*"
        )
    elif "BUY" in sig or "BULLISH" in reg:
        st.success(f"### Live Signal ({latest_time})\n# {sig}")
    elif "SELL" in sig or "BEARISH" in reg:
        st.error(f"### Live Signal ({latest_time})\n# {sig}")
    else:
        st.info(f"### Live Signal ({latest_time})\n# {sig}")

with col_s2:
    summary_data = {
        "Custom Formula Column": [
            "A: 5M Close (1-5 Channel)",
            "B: Kalman on A (Q = 0.50)",
            "C: Hurst of B",
            "D: Weighted Momentum (A - B)",
            "E: Kalman Filter (Q=0.50) on (C*D*1000)",
            "Supertrend (7,3) on Filtered E",
        ],
        "Live Value": [
            f"{latest['5M_Col_A_Channel']:.3f}",
            f"{latest['5M_Col_B_Kalman']:.3f}",
            f"{latest['5M_Col_C_Hurst']:.3f}",
            f"{latest['5M_Col_D_Weighted_Momentum']:.4f}",
            f"{latest['5M_Col_E_Composite']:.2f}",
            f"{latest['5M_Supertrend_on_E']:.2f}",
        ],
    }
    st.table(pd.DataFrame(summary_data))

st.markdown("---")

st.subheader("📋 Nifty 50 Timeline Grid (Kalman 0.50 on Scaled E)")

clean_cols = [
    "Market_Regime",
    "5M_HA_Close",
    "5M_Col_A_Channel",
    "5M_Col_B_Kalman",
    "5M_Col_C_Hurst",
    "5M_Col_D_Weighted_Momentum",
    "5M_Col_E_Composite",
    "5M_Supertrend_on_E",
    "Kinematic_Signal",
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(
    columns={
        "Market_Regime": "Regime",
        "5M_HA_Close": "5M Price",
        "5M_Col_A_Channel": "A (1-5 Channel)",
        "5M_Col_B_Kalman": "B (Kalman Q=0.50)",
        "5M_Col_C_Hurst": "C (Hurst of B)",
        "5M_Col_D_Weighted_Momentum": "D (Momentum A-B)",
        "5M_Col_E_Composite": "E (Kalman Q=0.50)",
        "5M_Supertrend_on_E": "Supertrend(E)",
        "Kinematic_Signal": "Kinematic Signal",
    },
    inplace=True,
)

for c in [
    "5M Price",
    "A (1-5 Channel)",
    "B (Kalman Q=0.50)",
    "C (Hurst of B)",
    "D (Momentum A-B)",
    "E (Kalman Q=0.50)",
    "Supertrend(E)",
]:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(display_df, use_container_width=True, height=650)
