import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Nifty 50 Speed & Chaos Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚀 NIFTY 50 Market Speed, Acceleration & Chaos Engine")
st.caption(
    "Live Physics-Based Velocity, Acceleration (G-Force), Lyapunov Chaos & Heikin-Ashi Signals"
)


# =====================================================================
# KINEMATIC & CHAOS MATHEMATICAL ENGINES
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


def compute_lyapunov_chaos_exponent(series, window=30):
    s = pd.Series(series)
    returns = np.abs(np.log(s / s.shift(1)).fillna(0.0).to_numpy())

    lle = np.full(len(series), 0.5)
    for i in range(window, len(returns)):
        sub = returns[i - window + 1 : i + 1]
        std_val = np.std(sub)
        mean_val = np.mean(sub) + 1e-8
        divergence = std_val / mean_val
        lle[i] = np.clip(divergence / (1.0 + divergence), 0.0, 1.0)

    return lle


def compute_fractal_dimension(series, window=30):
    s = pd.Series(series).to_numpy()
    df_val = np.full(len(series), 1.5)

    for i in range(window, len(series)):
        sub = s[i - window + 1 : i + 1]
        max_p = np.max(sub)
        min_p = np.min(sub)

        if max_p - min_p < 1e-8:
            df_val[i] = 1.0
            continue

        norm_sub = (sub - min_p) / (max_p - min_p)
        path_length = np.sum(
            np.sqrt(np.diff(norm_sub) ** 2 + (1.0 / (window - 1)) ** 2)
        )
        d = 1.0 + (np.log(path_length) + np.log(2.0)) / np.log(2.0 * (window - 1))
        df_val[i] = np.clip(d, 1.0, 2.0)

    return df_val


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
    raw_close = df_ha["Close"].to_numpy().flatten()

    # Physics Metrics: Speed (Velocity) & Acceleration
    window = 5
    df_ha["Velocity_Speed"] = df_ha["Close"].diff(window) / window
    df_ha["Acceleration_GForce"] = df_ha["Velocity_Speed"].diff(window) / window
    volatility = df_ha["Close"].rolling(window).std()
    df_ha["Kinetic_Force"] = df_ha["Velocity_Speed"] * volatility

    df_ha["Hurst"] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    df_ha["Lyapunov_Chaos"] = compute_lyapunov_chaos_exponent(
        raw_close, window=30
    )
    df_ha["Fractal_Dimension"] = compute_fractal_dimension(raw_close, window=30)

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


with st.spinner("Fetching Extended Nifty 50 Data & Calculating Speed Vectors..."):
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
# MULTI-TIMEFRAME ALIGNMENT
# =====================================================================
df_15m_grid = df_15m.copy()

# Shift inputs by 1 bar for leak-free accuracy
df_15m_grid["1H_HA_Close_Frozen"] = (
    df_1h["HA_Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["HA_HAM_1H_Frozen"] = (
    df_1h["HA_HAM"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["HA_HAM_1H_Prev"] = (
    df_1h["HA_HAM"].shift(2).reindex(df_15m_grid.index, method="ffill")
)

df_15m_grid["5M_Close"] = (
    df_5m["Close"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Velocity"] = (
    df_5m["Velocity_Speed"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Acceleration"] = (
    df_5m["Acceleration_GForce"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Lyapunov_Chaos"] = (
    df_5m["Lyapunov_Chaos"].shift(1).reindex(df_15m_grid.index, method="ffill")
)
df_15m_grid["5M_Fractal_Dimension"] = (
    df_5m["Fractal_Dimension"].shift(1).reindex(df_15m_grid.index, method="ffill")
)

n = len(df_15m_grid)
h1_curr_arr = df_15m_grid["HA_HAM_1H_Frozen"].to_numpy()
h1_prev_arr = df_15m_grid["HA_HAM_1H_Prev"].to_numpy()
m15_curr_arr = df_15m_grid["HA_HAM"].to_numpy()

ha_close_vals = df_15m_grid["HA_Close"].to_numpy()
ha_open_vals = df_15m_grid["HA_Open"].to_numpy()

hurst_vals = df_15m_grid["Hurst"].to_numpy()
chaos_5m_vals = df_15m_grid["5M_Lyapunov_Chaos"].to_numpy()
fractal_5m_vals = df_15m_grid["5M_Fractal_Dimension"].to_numpy()
vel_5m_vals = df_15m_grid["5M_Velocity"].to_numpy()
acc_5m_vals = df_15m_grid["5M_Acceleration"].to_numpy()

signals = ["⚪ NEUTRAL"] * n
regime = ["⚡ ORDERED / TRENDING"] * n
speed_status = ["🚀 REGULAR"] * n

for i in range(2, n):
    h1_curr = h1_curr_arr[i]
    h1_prev = h1_prev_arr[i]
    m15_curr = m15_curr_arr[i]
    h_val = hurst_vals[i]
    ch_5m = chaos_5m_vals[i]
    f_5m = fractal_5m_vals[i]
    v_5m = vel_5m_vals[i]
    a_5m = acc_5m_vals[i]

    is_ha_red = ha_close_vals[i] < ha_open_vals[i]

    # Speed & Acceleration Logic
    if v_5m > 2.0 and a_5m > 0:
        speed_status[i] = "⚡ ROCKET ACCELERATION UP"
    elif v_5m < -2.0 and a_5m < 0:
        speed_status[i] = "🔥 FAST DROP ACCELERATION"
    elif abs(v_5m) > 2.0 and abs(a_5m) < 0.1:
        speed_status[i] = "⚠️ BRAKES APPLIED (Speed Peak)"
    else:
        speed_status[i] = "🐢 NORMAL / CHOP SPEED"

    # Chaos Rule
    if ch_5m > 0.65 or f_5m > 1.55 or h_val < 0.48:
        regime[i] = "🌀 TURBULENT CHAOS ZONE"
        signals[i] = "🛑 NO TRADE (Choppy Market)"
        continue

    regime[i] = "⚡ ORDERED / TRENDING"

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
    elif h1_curr > h1_prev and h1_curr > 0:
        signals[i] = "🟢 ACCELERATED RALLY 🚀"
    elif h1_curr < h1_prev and h1_curr < 0:
        signals[i] = "🔴 ACCELERATED DROP 🔥"

df_15m_grid["Market_Regime"] = regime
df_15m_grid["Kinematic_Signal"] = signals
df_15m_grid["Speed_State"] = speed_status

df_15m_grid.dropna(inplace=True)
latest = df_15m_grid.iloc[-1]
latest_time = df_15m_grid.index[-1].strftime("%Y-%m-%d %H:%M IST")

# =====================================================================
# LIVE SPEEDOMETER & DASHBOARD DISPLAY
# =====================================================================
st.markdown("---")
st.subheader("🏎️ Live Market Speed & Acceleration Radar")

m_c1, m_c2, m_c3, m_c4 = st.columns(4)

with m_c1:
    v_val = latest["5M_Velocity"]
    st.metric(
        label="5M Market Speed (Velocity)",
        value=f"{v_val:+.2f} Pts/Bar",
        delta="Fast Move" if abs(v_val) > 2 else "Normal",
    )

with m_c2:
    a_val = latest["5M_Acceleration"]
    st.metric(
        label="5M Market G-Force (Acceleration)",
        value=f"{a_val:+.2f}",
        delta="Accelerating" if a_val > 0 else "Decelerating",
    )

with m_c3:
    st.metric(
        label="Speed State",
        value=f"{latest['Speed_State']}",
    )

with m_c4:
    st.metric(
        label="Lyapunov Chaos",
        value=f"{latest['5M_Lyapunov_Chaos']:.2f}",
        delta="Safe" if latest["5M_Lyapunov_Chaos"] < 0.65 else "High Chaos",
        delta_color="inverse",
    )

st.markdown("---")

col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    sig = latest["Kinematic_Signal"]
    reg = latest["Market_Regime"]

    if reg == "🌀 TURBULENT CHAOS ZONE":
        st.warning(
            f"### Market State: {reg}\n# 🛑 AVOID TRADING\n*(High Chaos / Low Hurst)*"
        )
    elif "RALLY" in sig or "BOTTOM" in sig:
        st.success(f"### Signal ({latest_time})\n# {sig}")
    elif "DROP" in sig or "TOP" in sig:
        st.error(f"### Signal ({latest_time})\n# {sig}")
    else:
        st.info(f"### Signal ({latest_time})\n# {sig}")

with col_s2:
    summary_table = pd.DataFrame(
        {
            "Kinematic Metric": [
                "15M Price",
                "5M Speed (Velocity Pts/Bar)",
                "5M Acceleration (G-Force)",
                "Hurst Exponent",
                "Chaos Exponent",
            ],
            "Live Value": [
                f"{latest['HA_Close']:,.2f}",
                f"{latest['5M_Velocity']:+.2f}",
                f"{latest['5M_Acceleration']:+.2f}",
                f"{latest['Hurst']:.2f}",
                f"{latest['5M_Lyapunov_Chaos']:.2f}",
            ],
            "Interpretation": [
                "Current Level",
                "High Speed > |2.0|",
                "> 0 Acceleration / < 0 Brake",
                "> 0.5 Trend / < 0.5 Chop",
                "< 0.65 Safe / > 0.65 Chaos",
            ],
        }
    )
    st.table(summary_table)

st.markdown("---")
st.subheader("📋 Nifty 50 Speed & Signal Timeline")

clean_cols = [
    "Market_Regime",
    "Speed_State",
    "5M_Velocity",
    "5M_Acceleration",
    "5M_Lyapunov_Chaos",
    "Hurst",
    "HA_Close",
    "Kinematic_Signal",
]
display_df = df_15m_grid[clean_cols].copy()

display_df.rename(
    columns={
        "Market_Regime": "Regime",
        "Speed_State": "Speed Status",
        "5M_Velocity": "5M Speed",
        "5M_Acceleration": "5M Acceleration",
        "5M_Lyapunov_Chaos": "Chaos",
        "Hurst": "Hurst",
        "HA_Close": "15M HA-Close",
        "Kinematic_Signal": "Signal",
    },
    inplace=True,
)

for c in ["5M Speed", "5M Acceleration", "Chaos", "Hurst", "15M HA-Close"]:
    display_df[c] = display_df[c].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

st.dataframe(display_df, use_container_width=True, height=600)
