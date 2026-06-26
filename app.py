import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OU Process Matrix Tight", layout="wide")

st.title("🏛️ Institutional Mean Reversion — Ornstein-Uhlenbeck (OU) Matrix")
st.write(
    "Process Noise Multiplier for **b (Kalman)** is locked at **0.0001** for an ultra-smooth, stable long-term target benchmark."
)

# ==========================================
# 1. FIXED DATE DATA LOCK PIPELINE (NSE TIMELINE)
# ==========================================


@st.cache_data(ttl=86400)
def generate_exact_nse_data():
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-06-26")

    raw_dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    nse_dates = [
        dt for dt in raw_dates if dt.weekday() < 5 and 9 <= dt.hour <= 15
    ]

    total_candles = len(nse_dates)
    base_close = 24000
    np.random.seed(12345)

    mock_history = base_close + np.cumsum(
        np.random.normal(0, 15, total_candles)
    )

    df_out = pd.DataFrame({"a": mock_history}, index=nse_dates)
    df_out.index.name = "Date & Time"
    return df_out


with st.spinner("⏳ Running OU Solvers with Tight Kalman Multiplier..."):
    df = generate_exact_nse_data()

    a_vals = df["a"].to_numpy().flatten()
    n = len(a_vals)

    # ==========================================
    # 2. CALCULATION: b (Standard Kalman on a with 0.0001 Multiplier)
    # ==========================================
    b_vals = np.zeros(n)
    x_b = a_vals[0]
    p_b = 1.0
    q_b = 0.0001  # 🔥 FIXED: Tight process noise multiplier as requested
    r_b = 1.0  # Measurement noise
    b_vals[0] = x_b

    for i in range(1, n):
        p_b = p_b + q_b
        k_gain = p_b / (p_b + r_b)
        x_b = x_b + k_gain * (a_vals[i] - x_b)
        p_b = (1 - k_gain) * p_b
        b_vals[i] = x_b

    df["b"] = b_vals
    df["c"] = df["a"] - df["b"]

    # ==========================================
    # 3. CALCULATION: ORNSTEIN-UHLENBECK PARAMS
    # ==========================================
    lookback = 30
    theta_arr = np.zeros(n)
    half_life_arr = np.zeros(n)
    ou_signal = np.zeros(n)

    c_vals = df["c"].to_numpy()

    for i in range(lookback, n):
        y = c_vals[i - lookback + 1 : i + 1]
        x = c_vals[i - lookback : i]

        poly = np.polyfit(x, y, 1)
        beta = poly[0]

        if 0 < beta < 1:
            theta = -np.log(beta)
            half_life = np.log(2) / theta
        else:
            theta = 0.01
            half_life = 99.0

        theta_arr[i] = theta
        half_life_arr[i] = half_life

        rolling_std = np.std(c_vals[i - lookback : i])
        ou_equilibrium_barrier = rolling_std * 1.8

        if c_vals[i] > ou_equilibrium_barrier and c_vals[i] < c_vals[i - 1]:
            ou_signal[i] = -1
        elif (
            c_vals[i] < -ou_equilibrium_barrier and c_vals[i] > c_vals[i - 1]
        ):
            ou_signal[i] = 1

    df["Theta"] = theta_arr
    df["Half_Life_Hours"] = half_life_arr
    df["Signal"] = ou_signal

st.success("📊 OU Model with Locked 0.0001 Multiplier Integrated!")

# ==========================================
# 4. STREAMLIT UI DISPLAY
# ==========================================
df_display = df.copy()
df_display.index = df_display.index.strftime("%Y-%m-%d %H:%M")

col1, col2, col3 = st.columns(3)
col1.metric(label="Kalman Q-Multiplier", value="0.0001")
col2.metric(label="Model Type", value="Stable OU Process")
col3.metric(label="Total Continuous Rows", value=f"{len(df_display)} Hours")

st.markdown("---")
st.subheader("🎯 Active OU Reversion Triggers (Strict High-Speed Only)")
sig_df = df_display[df_display["Signal"] != 0][
    ["a", "b", "c", "Theta", "Half_Life_Hours", "Signal"]
]
st.dataframe(sig_df.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")
st.subheader("📋 Full SDE Parameter Timeline Matrix")
st.dataframe(
    df_display[["a", "b", "c", "Theta", "Half_Life_Hours", "Signal"]].style.format(
        "{:.2f}"
    ),
    use_container_width=True,
    height=500,
)
