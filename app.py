from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="BTC-USD 1H Dual HAM Engine (Zero Future Leakage)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ BTC-USD 1H Strict Bar-Closed Dual HAM Engine")
st.caption(
    "100% Strict Freeze-on-Bar-Close Logic | 2-Year BTC-USD History (IST) | 50:50 Learn & Predict Split"
)


# =====================================================================
# 1. STRICT HEIKIN-ASHI CALCULATION (FREEZE ON BAR CLOSE)
# =====================================================================
def compute_heikin_ashi_strict(df_in: pd.DataFrame) -> pd.DataFrame:
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
# 2. CUSTOM CAUSAL KALMAN FILTER (ZERO BACKWARD SMOOTHING)
# =====================================================================
def apply_kalman_filter_causal(
    data_array, initial_p=50.0, q_val=0.0005, r_val=0.2
):
    if len(data_array) == 0:
        return np.array([])
    x, p = data_array[0], initial_p
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return np.array(filtered_values)


# =====================================================================
# 3. LEAK-FREE ROLLING HURST EXPONENT (STRICT LOOKBACK)
# =====================================================================
def calculate_rolling_hurst_leak_free(price_series, window=30):
    n = len(price_series)
    hurst_values = np.full(n, 0.5)
    s = pd.Series(price_series)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()

    for i in range(window, n):
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
# 4. STRICT DUAL HAM FEATURE GENERATOR (W-30 & W-200)
# =====================================================================
def compute_strict_btc_ham_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi_strict(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # Physics Velocity & Acceleration on Closed Bars
    window = 5
    df_ha["Velocity_Speed"] = df_ha["Close"].diff(window) / window
    df_ha["Acceleration_GForce"] = df_ha["Velocity_Speed"].diff(window) / window

    # 1. HAM Window 30 (Fast Momentum)
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(ha_close, window=30)
    kalman_30 = apply_kalman_filter_causal(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30 = apply_kalman_filter_causal(
        ha_close - kalman_30, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_30"] = mom_30 * (df_ha["Hurst_30"].to_numpy() * 2.0)

    # 2. HAM Window 200 (Macro Baseline)
    df_ha["Hurst_200"] = calculate_rolling_hurst_leak_free(
        ha_close, window=200
    )
    kalman_200 = apply_kalman_filter_causal(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_200 = apply_kalman_filter_causal(
        ha_close - kalman_200, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_200"] = mom_200 * (df_ha["Hurst_200"].to_numpy() * 2.0)

    # STRICT FREEZE: Shifted by 1 bar so unclosed candle has 0 impact
    df_ha["HAM_30_Frozen"] = df_ha["HA_HAM_30"].shift(1)
    df_ha["HAM_200_Frozen"] = df_ha["HA_HAM_200"].shift(1)
    df_ha["Hurst_30_Frozen"] = df_ha["Hurst_30"].shift(1)
    df_ha["Hurst_200_Frozen"] = df_ha["Hurst_200"].shift(1)

    return df_ha


# =====================================================================
# 5. DATA INGESTION (2 YEARS BTC-USD 1H DATA)
# =====================================================================
@st.cache_data(ttl=3600)
def load_btc_2year_1h_data():
    df_btc = yf.download(
        tickers="BTC-USD", period="730d", interval="1h", progress=False
    )

    if isinstance(df_btc.columns, pd.MultiIndex):
        df_btc.columns = df_btc.columns.get_level_values(0)

    df_btc = df_btc.dropna(subset=["Open", "High", "Low", "Close"])

    # Convert Timezone strictly to IST (Asia/Kolkata)
    if df_btc.index.tz is None:
        df_btc.index = df_btc.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df_btc.index = df_btc.index.tz_convert("Asia/Kolkata")

    return df_btc


with st.spinner(
    "📥 Fetching 2 Years of BTC-USD 1H Data & Computing Freeze Engine..."
):
    try:
        df_btc_raw = load_btc_2year_1h_data()
    except Exception as e:
        st.error(f"🚨 Data Error: {e}")
        st.stop()

# Compute HAM Features
df_btc_processed = compute_strict_btc_ham_features(df_btc_raw)

# Cleaning & Backfill Verification
df_btc_clean = df_btc_processed.dropna(
    subset=["HAM_30_Frozen", "HAM_200_Frozen"]
).copy()
df_btc_clean = df_btc_clean.bfill()

# =====================================================================
# 6. 50:50 LEARN & PREDICTION SPLIT ENGINE
# =====================================================================
total_bars = len(df_btc_clean)
split_idx = int(total_bars * 0.50)

df_learn = df_btc_clean.iloc[:split_idx].copy()
df_predict = df_btc_clean.iloc[split_idx:].copy()

# Latest Frozen Bar State
latest_bar = df_btc_clean.iloc[-1]
latest_time = df_btc_clean.index[-1].strftime("%Y-%m-%d %H:%M IST")

# =====================================================================
# 7. DASHBOARD & DISPLAY
# =====================================================================
st.markdown("---")
st.subheader("📌 BTC-USD 1H Live Frozen State (Zero Live Candle Leakage)")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="BTC-USD Price (Closed Bar)",
        value=f"${latest_bar['Close']:,.2f}",
    )

with m2:
    st.metric(
        label="Frozen 1H HAM (Window 30)",
        value=f"{latest_bar['HAM_30_Frozen']:+.2f}",
        delta="Positive" if latest_bar["HAM_30_Frozen"] > 0 else "Negative",
    )

with m3:
    st.metric(
        label="Frozen 1H HAM (Window 200)",
        value=f"{latest_bar['HAM_200_Frozen']:+.2f}",
        delta="Macro Bull" if latest_bar["HAM_200_Frozen"] > 0 else "Macro Bear",
    )

with m4:
    st.metric(
        label="Frozen Hurst (W-30 / W-200)",
        value=f"{latest_bar['Hurst_30_Frozen']:.2f} / {latest_bar['Hurst_200_Frozen']:.2f}",
        delta="Trending" if latest_bar["Hurst_30_Frozen"] >= 0.50 else "Choppy",
    )

st.markdown("---")
st.subheader("📊 50:50 Dataset Partition Audit")

s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    st.info(f"**Total 1H Bars Processed:** {total_bars:,} Candles (~2 Years)")
with s_col2:
    st.success(
        f"**50% Learning Set (In-Sample):** {len(df_learn):,} Bars\n"
        f"*(From {df_learn.index[0].strftime('%d %b %Y')} to {df_learn.index[-1].strftime('%d %b %Y')})*"
    )
with s_col3:
    st.warning(
        f"**50% Prediction Set (Out-of-Sample):** {len(df_predict):,} Bars\n"
        f"*(From {df_predict.index[0].strftime('%d %b %Y')} to {df_predict.index[-1].strftime('%d %b %Y')})*"
    )

st.markdown("---")
tab1, tab2 = st.tabs(
    ["🔮 Out-of-Sample Prediction Set (50%)", "📚 In-Sample Learning Set (50%)"]
)

display_cols = [
    "Close",
    "HA_Close",
    "HAM_30_Frozen",
    "HAM_200_Frozen",
    "Hurst_30_Frozen",
    "Hurst_200_Frozen",
]

col_renames = {
    "Close": "BTC Close ($)",
    "HA_Close": "HA Close ($)",
    "HAM_30_Frozen": "Frozen HAM (W-30)",
    "HAM_200_Frozen": "Frozen HAM (W-200)",
    "Hurst_30_Frozen": "Frozen Hurst (W-30)",
    "Hurst_200_Frozen": "Frozen Hurst (W-200)",
}

with tab1:
    st.markdown(
        "#### ⚡ Prediction Phase (Out-of-Sample - 100% Strict Causal Freeze)"
    )
    p_df = df_predict[display_cols].copy().iloc[::-1]
    p_df.rename(columns=col_renames, inplace=True)
    p_df.index = p_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(p_df.round(2), use_container_width=True, height=500)

with tab2:
    st.markdown("#### 📚 Learning Phase (In-Sample Calibration Set)")
    l_df = df_learn[display_cols].copy().iloc[::-1]
    l_df.rename(columns=col_renames, inplace=True)
    l_df.index = l_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(l_df.round(2), use_container_width=True, height=500)
