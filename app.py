from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="BTC-USD 1H Quantum HAM Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚛️ BTC-USD 1H Quantum Mechanics + HAM Engine")
st.caption(
    "Quantum Wavefunction Probability Density | 7x Cascaded 30 EMA | Strict Bar Freeze"
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
# 2. CAUSAL KALMAN FILTER
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
# 3. QUANTUM WAVEFUNCTION & PROBABILITY DENSITY ENGINE
# =====================================================================
def compute_quantum_wavefunction(price_series, window=30):
    """Calculates Quantum Wave Probability State |Psi|^2 based on Harmonic Oscillator Potential"""
    s = pd.Series(price_series)
    mean_p = s.rolling(window).mean()
    std_p = s.rolling(window).std().fillna(1.0)

    # Normalized Position Displacement (x)
    x = (s - mean_p) / (std_p + 1e-8)

    # Quantum Harmonic Oscillator Wavefunction Ground State: Psi(x) ~ exp(-x^2 / 2)
    psi = np.exp(-(x**2) / 2.0)

    # Quantum Probability Density: |Psi|^2
    prob_density = psi**2

    # Quantum State Momentum Direction (+1 for Upper Quantum State, -1 for Lower)
    quantum_state = np.where(
        x >= 0, prob_density, -prob_density
    )  # Quantum Phase

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
# 5. STRICT DUAL HAM + QUANTUM FEATURE GENERATOR
# =====================================================================
def compute_strict_btc_quantum_ham(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_ha = compute_heikin_ashi_strict(df_raw)

    # 1. 7x Cascaded 30 EMA Base
    ema_lvl = df_ha["HA_Close"].copy()
    for _ in range(7):
        ema_lvl = ema_lvl.ewm(span=30, adjust=False).mean()

    df_ha["HA_Close_EMA30_L7"] = ema_lvl
    ha_close_ema30_l7 = df_ha["HA_Close_EMA30_L7"].to_numpy().flatten()
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # 2. HAM Fast Calculation
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(
        ha_close_ema30_l7, window=30
    )
    kalman_30_l7 = apply_kalman_filter_causal(
        ha_close_ema30_l7, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30_l7 = apply_kalman_filter_causal(
        ha_close_ema30_l7 - kalman_30_l7, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    df_ha["HA_HAM_30_EMA7"] = mom_30_l7 * (df_ha["Hurst_30"].to_numpy() * 2.0)

    # 3. HAM Macro 200
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

    # 4. Quantum Wavefunction Calculation
    df_ha["Quantum_State"] = compute_quantum_wavefunction(
        ha_close_ema30_l7, window=30
    )

    # STRICT FREEZE ON BAR CLOSE (Shifted by 1 bar for ZERO live leakage)
    df_ha["HAM_30_EMA7_Frozen"] = df_ha["HA_HAM_30_EMA7"].shift(1)
    df_ha["HAM_200_Frozen"] = df_ha["HA_HAM_200"].shift(1)
    df_ha["Quantum_State_Frozen"] = df_ha["Quantum_State"].shift(1)

    # Difference Spread
    df_ha["HAM_Spread"] = df_ha["HAM_30_EMA7_Frozen"] - df_ha["HAM_200_Frozen"]

    return df_ha


# =====================================================================
# 6. DATA INGESTION & RUN ENGINE
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

# Compute Features
df_btc_processed = compute_strict_btc_quantum_ham(df_btc_raw)
df_btc_clean = df_btc_processed.dropna(
    subset=["HAM_30_EMA7_Frozen", "Quantum_State_Frozen"]
).copy()
df_btc_clean = df_btc_clean.bfill()

# 50:50 Split
total_bars = len(df_btc_clean)
split_idx = int(total_bars * 0.50)
df_learn = df_btc_clean.iloc[:split_idx].copy()
df_predict = df_btc_clean.iloc[split_idx:].copy()

latest_bar = df_btc_clean.iloc[-1]

# =====================================================================
# 7. DASHBOARD DISPLAY
# =====================================================================
st.markdown("---")
st.subheader("📌 Live State: Quantum Wave Probability + Dual HAM")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="BTC Price (Last Closed)", value=f"${latest_bar['Close']:,.2f}"
    )
with m2:
    st.metric(
        label="Frozen HAM (30 EMA x7)",
        value=f"{latest_bar['HAM_30_EMA7_Frozen']:+.2f}",
    )
with m3:
    st.metric(
        label="Quantum Phase State (|Ψ|²)",
        value=f"{latest_bar['Quantum_State_Frozen']:+.4f}",
        delta=(
            "High Bullish State"
            if latest_bar["Quantum_State_Frozen"] > 0
            else "High Bearish State"
        ),
    )
with m4:
    st.metric(
        label="HAM Spread (30x7 - 200)", value=f"{latest_bar['HAM_Spread']:+.2f}"
    )

st.markdown("---")
tab1, tab2 = st.tabs(
    ["🔮 50% Prediction Set (Out-of-Sample)", "📚 50% Learning Set (In-Sample)"]
)

display_cols = [
    "Close",
    "HA_Close",
    "HAM_30_EMA7_Frozen",
    "Quantum_State_Frozen",
    "HAM_200_Frozen",
    "HAM_Spread",
]
col_renames = {
    "Close": "BTC Close ($)",
    "HA_Close": "HA Close ($)",
    "HAM_30_EMA7_Frozen": "Frozen HAM (30 EMA x7)",
    "Quantum_State_Frozen": "Quantum Wave Phase (|Ψ|²)",
    "HAM_200_Frozen": "Frozen HAM (W-200)",
    "HAM_Spread": "HAM Spread",
}

with tab1:
    st.markdown("#### Out-of-Sample Prediction Table (Quantum HAM Integration)")
    p_df = df_predict[display_cols].copy().iloc[::-1]
    p_df.rename(columns=col_renames, inplace=True)
    p_df.index = p_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(p_df.round(4), use_container_width=True, height=500)

with tab2:
    st.markdown("#### In-Sample Learning Table")
    l_df = df_learn[display_cols].copy().iloc[::-1]
    l_df.rename(columns=col_renames, inplace=True)
    l_df.index = l_df.index.strftime("%Y-%m-%d %H:%M IST")
    st.dataframe(l_df.round(4), use_container_width=True, height=500)
