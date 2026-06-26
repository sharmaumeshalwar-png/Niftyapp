import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Page layout configuration
st.set_page_config(page_title="Correct Kalman Matrix", layout="wide")

st.title("📋 Corrected Kalman Filter Matrix (Exact NSE Dates & Hours)")
st.write(
    "Yeh table **01-Jan-2025** se strictly shuru ho rahi hai, jisme sirf Indian Market hours aur weekdays locked hain."
)

# ==========================================
# 1. FIXED DATE DATA LOCK PIPELINE (NSE TIME HOURS LOCK)
# ==========================================


@st.cache_data(ttl=86400)
def generate_exact_nse_data():
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-06-26")  # Current Date Till Now

    # Pure Calendar Days Generator
    raw_dates = pd.date_range(start=start_date, end=end_date, freq="1h")

    # STRICT FILTERING: Sirf Weekdays (Mon-Fri) aur Market Hours (09:00 se 15:00) lock karna
    nse_dates = [
        dt
        for dt in raw_dates
        if dt.weekday() < 5 and 9 <= dt.hour <= 15  # Mon-Fri Only  # Market Time Hours
    ]

    total_candles = len(nse_dates)

    base_close = 24000
    np.random.seed(12345)  # Seed locked for absolute consistency

    # Vectorized fast sequence generation
    mock_history = base_close + np.cumsum(
        np.random.normal(0, 15, total_candles)
    )

    # Creating Dataframe with string formatted exact timestamps
    df_out = pd.DataFrame({"a": mock_history}, index=nse_dates)
    df_out.index.name = "Date & Time"
    return df_out


with st.spinner("⏳ Calculations running for exact NSE timestamps..."):
    df = generate_exact_nse_data()

    # Arrays for processing
    a_vals = df["a"].to_numpy().flatten()
    n = len(a_vals)

    # ==========================================
    # 2. CALCULATION: b (Standard Kalman on a)
    # ==========================================
    b_vals = np.zeros(n)
    x_b = a_vals[0]
    p_b = 1.0
    q_b = 0.01
    r_b = 1.0
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
    d_vals = np.zeros(n)
    x_d = c_vals[0]
    p_d = 1.0
    q_d_base = 0.05
    r_d = 1.0
    d_vals[0] = x_d

    c_series = pd.Series(c_vals)
    c_volatility = (
        c_series.rolling(window=14, min_periods=1).std().fillna(1.0).to_numpy()
    )

    for i in range(1, n):
        adaptive_q = q_d_base * (1.0 + abs(c_vals[i] / c_volatility[i]))
        p_d = p_d + adaptive_q
        k_gain_d = p_d / (p_d + r_d)
        x_d = x_d + k_gain_d * (c_vals[i] - x_d)
        p_d = (1 - k_gain_d) * p_d
        d_vals[i] = x_d

    df["d"] = d_vals

st.success("📊 Matrix Generation Complete with Correct Timeline!")

# ==========================================
# 5. STREAMLIT UI DISPLAY (PURE TABLE GRID)
# ==========================================
# ERROR FIXED HERE
df.index = df.index.strftime("%Y-%m-%d %H:%M")
final_display_df = df[["a", "b", "c", "d"]]

col1, col2, col3 = st.columns(3)
col1.metric(label="Data Locked From", value="01-Jan-2025 09:00")
col2.metric(label="Total Valid NSE Rows", value=f"{len(final_display_df)} Rows")

# CSV download hook with accurate formatting
csv_data = final_display_df.to_csv().encode("utf-8")
col3.download_button(
    label="📥 Download Corrected Data CSV",
    data=csv_data,
    file_name="nifty_perfect_time_kalman.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("📋 Core Mathematical Matrix Sequence (Scroll to see All Rows)")

# Render whole data sheet with proper precision
st.dataframe(
    final_display_df.style.format("{:.2f}"),
    use_container_width=True,
    height=600,
)
