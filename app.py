import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Page layout configuration
st.set_page_config(page_title="Kalman Matrix", layout="wide")

st.title("📋 Kalman Filter Core Matrix (Locked from Jan 2025)")
st.write(
    "Showing strictly columns: **a (Close)**, **b (Kalman)**, **c (a-b)**, and **d (Adaptive Kalman on c)**."
)

# ==========================================
# 1. FIXED 2-YEAR DATA PIPELINE (LOCKED SEED)
# ==========================================


@st.cache_data(ttl=86400)
def generate_frozen_nifty_data():
    start_date = pd.Timestamp("2025-01-01")
    # Exact 2 years timeline lock (Approx 17,520 hours)
    end_date = start_date + datetime.timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    total_candles = len(dates)

    base_close = 24000
    np.random.seed(12345)  # Seed frozen for exact historical consistency

    # Vectorized fast generation
    mock_history = base_close + np.cumsum(
        np.random.normal(0, 15, total_candles)
    )

    df_out = pd.DataFrame({"a": mock_history}, index=dates)
    return df_out


with st.spinner("⏳ Calculations running... Please wait"):
    df = generate_frozen_nifty_data()

    # Arrays for processing
    a_vals = df["a"].to_numpy().flatten()
    n = len(a_vals)

    # ==========================================
    # 2. CALCULATION: b (Standard Kalman on a)
    # ==========================================
    b_vals = np.zeros(n)
    x_b = a_vals[0]
    p_b = 1.0
    q_b = 0.01  # Fixed process noise
    r_b = 1.0  # Fixed measurement noise
    b_vals[0] = x_b

    for i in range(1, n):
        p_b = p_b + q_b
        k_gain = p_b / (p_b + r_b)
        x_b = x_b + k_gain * (a_vals[i] - x_b)
        p_b = (1 - k_gain) * p_b
        b_vals[i] = x_b

    df["b"] = b_vals

    # ==========================================
    # 3. CALCULATION: c (a - b)
    # ==========================================
    df["c"] = df["a"] - df["b"]
    c_vals = df["c"].to_numpy().flatten()

    # ==========================================
    # 4. CALCULATION: d (Adaptive Kalman on c)
    # ==========================================
    # Velocity/Volatility logic based on 'c' to make it adaptive
    d_vals = np.zeros(n)
    x_d = c_vals[0]
    p_d = 1.0
    q_d_base = 0.05
    r_d = 1.0
    d_vals[0] = x_d

    # Simple Rolling Deviation of 'c' for adaptive noise scaling
    c_series = pd.Series(c_vals)
    c_volatility = (
        c_series.rolling(window=14, min_periods=1).std().fillna(1.0).to_numpy()
    )

    for i in range(1, n):
        # Adaptive calculation: Volatility badhne par processing speed (Q) badhegi
        adaptive_q = q_d_base * (1.0 + abs(c_vals[i] / c_volatility[i]))

        p_d = p_d + adaptive_q
        k_gain_d = p_d / (p_d + r_d)
        x_d = x_d + k_gain_d * (c_vals[i] - x_d)
        p_d = (1 - k_gain_d) * p_d
        d_vals[i] = x_d

    df["d"] = d_vals

st.success("📊 Matrix Generation Complete!")

# ==========================================
# 5. STREAMLIT UI DISPLAY (PURE TABLE GRID)
# ==========================================
col1, col2 = st.columns(2)
col1.metric(label="Data Locked From", value="01-Jan-2025")
col2.metric(label="Total Frozen Rows", value=f"{len(df)} Candles (2 Years)")

st.markdown("---")
st.subheader("📋 Core Mathematical Matrix Sequence")
st.write(
    "Niche table mein columns ka matlab hai: **a** = Close Price, **b** = Kalman Line, **c** = Difference ($a-b$), **d** = Adaptive Kalman of $c$."
)

# Showing the clean table format with specific precision
final_display_df = df[["a", "b", "c", "d"]]

# Multi-index or extra headers are dropped, pure a, b, c, d is displayed
st.dataframe(
    final_display_df.tail(100).style.format("{:.2f}"),
    use_container_width=True,
)
