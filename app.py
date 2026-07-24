from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="BTC-USD 1H Quantum Engine + Correlation Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚛️ BTC-USD 1H Quantum Engine (with Auto-Correlation Filter)")
st.caption(
    "10,000x Scaled Quantum Wave Phase | Dynamic Rolling Correlation State | Zero Future Leakage (Shift 1)"
)


# =====================================================================
# 1. STRICT HEIKIN-ASHI CALCULATION
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
# 2. CAUSAL KALMAN FILTER (ZERO LEAKAGE)
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
# 3. QUANTUM WAVEFUNCTION ENGINE (10,000x MULTIPLIER)
# =====================================================================
def compute_quantum_wavefunction_10k(
    price_series, window=30, scale_factor=10000.0
):
    s = pd.Series(price_series)
    mean_p = s.rolling(window).mean()
    std_p = s.rolling(window).std().fillna(1.0)

    # Normalized Position Displacement (x)
    x = (s - mean_p) / (std_p + 1e-8)

    # Quantum Harmonic Oscillator Ground State: Psi(x) ~ exp(-x^2 / 2)
    psi = np.exp(-(x**2) / 2.0)
    prob_density = psi**2

    # Scaled Quantum Phase State Direction
    quantum_state = (
        np.where(x >= 0, prob_density, -prob_density) * scale_factor
    )
    return quantum_state


# =====================================================================
# 4. LEAK-FREE ROLLING HURST EXPONENT
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
# 5. METHOD 2: DYNAMIC ROLLING CORRELATION TRACKER
# =====================================================================
def compute_rolling_correlation_filter(s1: pd.Series, s2: pd.Series, window=30):
    """Calculates rolling Pearson Correlation and assigns sync states."""
    r = s1.rolling(window).corr(s2).fillna(0.0)

    # Z-Score of rolling correlation to measure extreme deviation
    r_mean = r.rolling(window).mean()
    r_std = r.rolling(window).std().replace(0, 1e-8)
    r_zscore = (r - r_mean) / r_std

    # Correlation State Logic
    # Strong correlation when |r| >= 0.70
    states = np.where(
        np.abs(r) >= 0.70, "IN_CORRELATION", "CORRELATION_BREAKDOWN"
    )

    return r, r_zscore, states


# =====================================================================
# 6. DUAL HAM + QUANTUM + CORRELATION PIPELINE
# =====================================================================
def compute_strict_btc_quantum_ham(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi_strict(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # 1. Normal Fast HAM 30
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(
        ha_close, window=30
    )
    kalman_30 = apply_kalman_filter_causal(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30 = apply_kalman_filter_causal(
        ha_close - kalman_30, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_30_Normal"] = mom_30 * (df_ha["Hurst_30"].to_numpy() * 2.0)

    # 2. Macro HAM 200
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

    # 3. Quantum Wavefunction 10,000x
    df_ha["Quantum_State"] = compute_quantum_wavefunction_10k(
        ha_close, window=30, scale_factor=10000.0
    )

    # 4. INTEGRATION OF METHOD 2: Dynamic Rolling Correlation (Close vs Quantum)
    r_val, r_zscore, corr_states = compute_rolling_correlation_filter(
        df_ha["Close"], df_ha["Quantum_State"], window=30
    )
    df_ha["Rolling_Corr"] = r_val
    df_ha["Corr_ZScore"] = r_zscore
    df_ha["Corr_State"] = corr_states

    # STRICT FREEZE ON BAR CLOSE (Shift 1 for Zero Future Leakage)
    df_ha["HAM_30_Normal_Frozen"] = df_ha["HA_HAM_30_Normal"].shift(1)
    df_ha["HAM_200_Frozen"] = df_ha["HA_HAM_200"].shift(1)
    df_ha["Quantum_State_Frozen"] = df_ha["Quantum_State"].shift(1)
    df_ha["Rolling_Corr_Frozen"] = df_ha["Rolling_Corr"].shift(1)
    df_ha["Corr_State_Frozen"] = df_ha["Corr_State"].shift(1)

    df_ha["HAM_Spread"] = (
        df_ha["HAM_30_Normal_Frozen"] - df_ha["HAM_200_Frozen"]
    )

    return df_ha


# =====================================================================
# 7. DATA INGESTION & ENGINE RUN
# =====================================================================
@st.cache_data(ttl=3600)
def load_btc_2year_1h_data():
    df_btc = yf.download(
        tickers="BTC-USD", period="730d", interval="1h", progress=False
    )
    if isinstance(df_btc.columns, pd.MultiIndex):
        df_btc.columns = df_btc.columns.get_level_values(0)
    df_btc = df_btc.dropna(subset=["Open", "High", "Low", "Close"])
    if df_btc.index.tz is None:
        df_btc.index = df_btc.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df_btc.index = df_btc.index.tz_convert("Asia/Kolkata")
    return df_btc


with st.spinner("📥 Fetching 2 Years BTC-USD 1H Data..."):
    try:
        df_btc_raw = load_btc_2year_1h_data()
    except Exception as e:
        st.error(f"🚨 Data Error: {e}")
        st.stop()

# Compute Pipeline
df_btc_processed = compute_strict_btc_quantum_ham(df_btc_raw)
df_btc_clean = df_btc_processed.dropna(
    subset=["HAM_30_Normal_Frozen", "Quantum_State_Frozen"]
).copy()
df_btc_clean = df_btc_clean.bfill()

# 50:50 Split
total_bars = len(df_btc_clean)
split_idx = int(total_bars * 0.50)
df_learn = df_btc_clean.iloc[:split_idx].copy()
df_predict = df_btc_clean.iloc[split_idx:].copy()

latest_bar = df_btc_clean.iloc[-1]

# =====================================================================
# 8. DASHBOARD DISPLAY
# =====================================================================
st.markdown("---")
st.subheader(
    "📌 Live State: 10,000x Quantum Phase + Rolling Correlation Filter"
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(
        label="BTC Price (Last Closed)", value=f"${latest_bar['Close']:,.2f}"
    )
with m2:
    st.metric(
        label="Frozen HAM (Normal 30)",
        value=f"{latest_bar['HAM_30_Normal_Frozen']:+.2f}",
    )
with m3:
    st.metric(
        label="Quantum Phase (|Ψ|² x10k)",
        value=f"{latest_bar['Quantum_State_Frozen']:+.2f}",
    )
with m4:
    st.metric(
        label="Rolling Corr (30 W)",
        value=f"{latest_bar['Rolling_Corr_Frozen']:+.2f}",
    )
with m5:
    corr_status = latest_bar["Corr_State_Frozen"]
    st.metric(
        label="Correlation State",
        value=corr_status,
        delta=(
            "Synced (Safe)"
            if corr_status == "IN_CORRELATION"
            else "Breakdown Alert!"
        ),
        delta_color="normal" if corr_status == "IN_CORRELATION" else "inverse",
    )

st.markdown("---")
tab1, tab2 = st.tabs(
    ["🔮 50% Prediction Set (Out-of-Sample)", "📚 50% Learning Set (In-Sample)"]
)

display_cols = [
    "Close",
    "HA_Close",
    "HAM_30_Normal_Frozen",
    "Quantum_State_Frozen",
    "Rolling_Corr_Frozen",
    "Corr_State_Frozen",
    "HAM_Spread",
]

col_renames = {
    "Close": "BTC Close ($)",
    "HA_Close": "HA Close ($)",
    "HAM_30_Normal_Frozen": "Frozen Fast HAM (W-30)",
    "Quantum_State_Frozen": "Quantum Phase (|Ψ|² x10k)",
    "Rolling_Corr_Frozen": "Rolling Corr (30 Close)",
    "Corr_State_Frozen": "Correlation Status",
    "HAM_Spread": "HAM Spread",
}

with tab1:
    st.markdown("#### Out-of-Sample Prediction Table (Method 2 Active)")
    p_df = df_predict[display_cols].copy().iloc[::-1]
    p_df.rename(columns=col_renames, inplace=True)
    p_df.index = p_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(p_df.round(2), use_container_width=True, height=500)

with tab2:
    st.markdown("#### In-Sample Learning Table")
    l_df = df_learn[display_cols].copy().iloc[::-1]
    l_df.rename(columns=col_renames, inplace=True)
    l_df.index = l_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(l_df.round(2), use_container_width=True, height=500)
