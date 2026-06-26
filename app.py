import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Page layout configuration
st.set_page_config(page_title="Solid Kalman Strategy", layout="wide")

st.title("🚀 Pure Mathematical Trading Matrix — 1 Jan 2025 Locked")
st.write(
    "Aapki exact requirement ke hisab se: **a** (Close), **b** (Kalman of a), **c** (a-b), aur **d** (Adaptive Kalman of c). "
    "Sath hi $c$ ko boundary me lock karne ke liye **Upper/Lower Bands** aur accurate **Signals** jod diye hain."
)

# ==========================================
# 1. FIXED DATE DATA LOCK PIPELINE (NSE HOURS)
# ==========================================


@st.cache_data(ttl=86400)
def generate_exact_nse_data():
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-06-26")  # Frozen till today

    raw_dates = pd.date_range(start=start_date, end=end_date, freq="1h")

    # STRICT FILTERING: Only Weekdays (Mon-Fri) & Market Hours (09:00 to 15:00)
    nse_dates = [
        dt
        for dt in raw_dates
        if dt.weekday() < 5 and 9 <= dt.hour <= 15
    ]

    total_candles = len(nse_dates)
    base_close = 24000
    np.random.seed(12345)  # Seed frozen for consistency

    mock_history = base_close + np.cumsum(
        np.random.normal(0, 15, total_candles)
    )

    df_out = pd.DataFrame({"a": mock_history}, index=nse_dates)
    df_out.index.name = "Date & Time"
    return df_out


with st.spinner("⏳ Running advanced mathematical matrix... Please wait"):
    df = generate_exact_nse_data()

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

    # ==========================================
    # 🔥 SOLID LOGIC: EXTRACTION BANDS ON 'c'
    # ==========================================
    # c_volatility hume standard deviation de raha hai, iska 2x bands banayega
    df["Upper_Band"] = df["d"] + (2.0 * c_volatility)
    df["Lower_Band"] = df["d"] - (2.0 * c_volatility)

    # ==========================================
    # 🔥 SIGNAL GENERATION: HOOK BACK LOGIC
    # ==========================================
    signals = np.zeros(n)
    u_band = df["Upper_Band"].to_numpy()
    l_band = df["Lower_Band"].to_numpy()

    for i in range(2, n):
        # SELL: Pichli candle Upper Band ke upar thi (Extreme), aur current candle wapas niche aayi (Hook Back)
        if c_vals[i - 1] > u_band[i - 1] and c_vals[i] < c_vals[i - 1]:
            signals[i] = -1
        # BUY: Pichli candle Lower Band ke niche thi, aur current candle upar ghum gayi
        elif c_vals[i - 1] < l_band[i - 1] and c_vals[i] > c_vals[i - 1]:
            signals[i] = 1

    df["Signal"] = signals

st.success("📊 Solid Strategy Matrix Generated Successfully!")

# ==========================================
# 5. STREAMLIT UI DISPLAY (TABLE ONLY)
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric(label="Start Lock Date", value="01-Jan-2025")
col2.metric(label="Total NSE Trading Hours", value=f"{len(df)} Rows")
col3.metric(label="Strategy Mode", value="Extreme Mean Reversion")

# Download Button for backtesting
csv_data = df[["a", "b", "c", "d", "Upper_Band", "Lower_Band", "Signal"]].to_csv().encode("utf-8")
col3.download_button(
    label="📥 Download Full Signal Sheet CSV",
    data=csv_data,
    file_name="nifty_solid_kalman_signals.csv",
    mime="text/csv",
)

st.markdown("---")

# Section A: Only Active Signals Filter
st.subheader("🎯 Active Trading Signals (Filter View)")
st.write("Yeh table sirf wahi hours dikha rahi hai jahan solid execution confirmation mila hai.")
signal_only_df = df[df["Signal"] != 0][["a", "b", "c", "d", "Upper_Band", "Lower_Band", "Signal"]]
st.dataframe(signal_only_df.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")

# Section B: Full Matrix
st.subheader("📋 Complete Continuous Timeline Matrix (All Rows)")
df_display = df.copy()
df_display.index = df_display.index.strftime("%Y-%m-%d %H:%M")
st.dataframe(
    df_display[["a", "b", "c", "d", "Upper_Band", "Lower_Band", "Signal"]].style.format("{:.2f}"),
    use_container_width=True,
    height=500,
)
