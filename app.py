import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Page layout configuration
st.set_page_config(page_title="Complete Kalman Matrix", layout="wide")

st.title("📋 Complete Kalman Filter Matrix (Full 2-Year Frozen Data)")
st.write(
    "Yeh table **01-Jan-2025** se lekar poora data show kar rahi hai. Aap isse niche scroll karke poora dekh sakte hain aur download bhi kar sakte hain."
)

# ==========================================
# 1. FIXED 2-YEAR DATA PIPELINE (FULL DATA DETECT)
# ==========================================


@st.cache_data(ttl=86400)
def generate_frozen_nifty_data():
    start_date = pd.Timestamp("2025-01-01")
    # Exact 2 years timeline lock (Approx 17,520 hours/rows)
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


with st.spinner("⏳ Calculations running for full dataset... Please wait"):
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

st.success("📊 Poora Data (All Rows) Successfully Calculate Ho Gaya Hai!")

# ==========================================
# 5. STREAMLIT UI DISPLAY (POORA DATA GRID)
# ==========================================
final_display_df = df[["a", "b", "c", "d"]]

col1, col2, col3 = st.columns(3)
col1.metric(label="Data Start Date", value="01-Jan-2025")
col2.metric(label="Total Rows in Table", value=f"{len(final_display_df)} Rows")

# Feature: Download Button taaki aap Excel me poora data nikal sako
csv_data = final_display_df.to_csv().encode("utf-8")
col3.download_button(
    label="📥 Download Full Data as CSV",
    data=csv_data,
    file_name="nifty_kalman_full_2025_2027.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("📋 Core Mathematical Matrix Sequence (Scroll Down to see All Rows)")

# FIX: `.tail()` ya `.head()` hata diya hai, ab poora data frame display hoga grid me height setting ke sath
st.dataframe(
    final_display_df.style.format("{:.2f}"),
    use_container_width=True,
    height=600,  # Isse table me scrollbar aa jayega aur crash nahi hoga
)
