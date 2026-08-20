import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="MCX Silver 2-Year 50:50 Kinematics Engine", layout="wide"
)
st.title("⚡ MCX Silver 2-Year Kinematics Engine (50:50 Dual Window)")
st.write(
    "🎯 **2-Year Timeframe Engine:** 50% In-Sample Historical Analysis | 50%"
    " Out-of-Sample Prediction Engine | **Zero-Repaint & Zero-Leakage Lock**"
)

# Sidebar Controls
st.sidebar.header("🔄 Engine Parameters")

gaussian_sigma = st.sidebar.slider(
    "🔔 Hubble Gaussian Sigma (σ)",
    min_value=0.5,
    max_value=20.0,
    value=10.0,
    step=0.5,
    help="Gaussian bell-curve smoothing applied to Zero-Centered Hubble Velocity.",
)

if st.sidebar.button("⚡ Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal Rolling Window)\n\n"
    "🔒 **Repaint Protection:** ZERO REPAINT (Historical Values Locked)\n\n"
    "📅 **Data Horizon:** 2 Years (1D Interval)\n\n"
    "⚖️ **Split Ratio:** 50% In-Sample : 50% Out-of-Sample"
)


# =====================================================================
# MATHEMATICAL ENGINES & FILTERS (STRICT CAUSAL - NO REPAINT)
# =====================================================================
def apply_kalman_filter_causal(data_array, initial_p=0.50, q_val=0.005, r_val=0.1):
    """Sequential Causal Kalman Filter."""
    arr = np.asarray(data_array, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([])
    x, p = arr[0], initial_p
    filtered_values = np.empty(len(arr))
    for i, z in enumerate(arr):
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values[i] = x
    return filtered_values


def apply_zlema_causal(data_array, period=10):
    """Strict Causal Zero-Lag Exponential Moving Average (ZLEMA)."""
    arr = np.asarray(data_array, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([])
    
    lag = int((period - 1) / 2)
    zlema_vals = np.empty_like(arr)
    
    de_lagged = np.empty_like(arr)
    for i in range(len(arr)):
        if i >= lag:
            de_lagged[i] = arr[i] + (arr[i] - arr[i - lag])
        else:
            de_lagged[i] = arr[i]
            
    alpha = 2.0 / (period + 1.0)
    ema = de_lagged[0]
    for i in range(len(arr)):
        ema = alpha * de_lagged[i] + (1.0 - alpha) * ema
        zlema_vals[i] = ema
        
    return zlema_vals


def apply_bipolar_wave_gaussian_causal(data_array, sigma=10.0, window_size=25):
    """
    Causal Bipolar Gaussian Wave Engine:
    - Zero-Centers values using past-only rolling mean
    - Applies Smooth Causal Gaussian Filter
    - 100% Causal, Zero-Repaint, Zero Future Leakage
    """
    arr = np.asarray(data_array, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([])
    
    smoothed = np.empty_like(arr)
    for i in range(len(arr)):
        start_idx = max(0, i - window_size)
        sub_arr = arr[start_idx : i + 1]
        
        if len(sub_arr) > 0:
            rolling_mean = np.mean(sub_arr)
            centered_sub_arr = sub_arr - rolling_mean
            
            filt = gaussian_filter1d(centered_sub_arr, sigma=sigma, mode="nearest")
            raw_gauss = filt[-1]
            
            lag_offset = max(1, int((sigma - 1.0) / 2.0))
            if len(sub_arr) > lag_offset:
                past_gauss = filt[-1 - lag_offset]
                zero_lag_val = 2.0 * raw_gauss - past_gauss
            else:
                zero_lag_val = raw_gauss
                
            smoothed[i] = zero_lag_val
        else:
            smoothed[i] = 0.0
            
    final_wave = apply_zlema_causal(smoothed, period=int(sigma))
    return final_wave


def calculate_rolling_hurst_causal(price_series, window=30):
    """Strict Causal Rolling Hurst Exponent."""
    arr = np.asarray(price_series, dtype=float).flatten()
    s = pd.Series(arr)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    hurst_values = np.full(len(arr), 0.5)

    for i in range(window, len(arr)):
        sub_returns = log_returns[i - window + 1 : i + 1]
        mean_val = np.mean(sub_returns)
        cum_dev = np.cumsum(sub_returns - mean_val)
        r_val = np.ptp(cum_dev)
        s_val = np.std(sub_returns, ddof=1) + 1e-10
        rs_ratio = r_val / s_val
        if rs_ratio > 0:
            h_val = np.log(rs_ratio) / np.log(window)
            hurst_values[i] = np.clip(h_val, 0.0, 1.0)
            
    return hurst_values


def apply_heikin_ashi(df_in):
    op = np.asarray(df_in["Open"], dtype=float).flatten()
    hi = np.asarray(df_in["High"], dtype=float).flatten()
    lo = np.asarray(df_in["Low"], dtype=float).flatten()
    cl = np.asarray(df_in["Close"], dtype=float).flatten()

    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0

    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))

    df_out = df_in.copy()
    df_out["HA_Open"] = ha_open
    df_out["HA_High"] = ha_high
    df_out["HA_Low"] = ha_low
    df_out["HA_Close"] = ha_close
    return df_out


def apply_hysteresis_state_machine(df_in, reversal_threshold_pct=0.20):
    df = df_in.copy()

    raw_velocity = df["HAM_Diff_Kalman"].diff().fillna(0.0).to_numpy()
    df["HAM_Velocity"] = apply_kalman_filter_causal(
        raw_velocity, initial_p=0.50, q_val=0.005, r_val=0.1
    )
    df["HAM_Acceleration"] = df["HAM_Velocity"].diff().fillna(0.0)

    diff_vals = df["HAM_Diff_Kalman"].to_numpy()
    states = []

    current_state = "🟡 INITIALIZING"
    peak_val = diff_vals[0]
    trough_val = diff_vals[0]

    for i in range(len(diff_vals)):
        val = diff_vals[i]

        if i == 0:
            states.append("🟡 INITIALIZING")
            continue

        if val > peak_val:
            peak_val = val
        if val < trough_val:
            trough_val = val

        peak_drop_trigger = peak_val - (
            abs(peak_val) * reversal_threshold_pct + 1.0
        )
        trough_rise_trigger = trough_val + (
            abs(trough_val) * reversal_threshold_pct + 1.0
        )

        if current_state in ["🟡 INITIALIZING", "🟢 STRONG BULLISH TREND"]:
            if val < peak_drop_trigger:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"
                trough_val = val
            else:
                current_state = "🟢 STRONG BULLISH TREND"

        elif current_state == "🔴 STRONG BEARISH TREND (Rally Stopped)":
            if val > trough_rise_trigger:
                current_state = "🟢 STRONG BULLISH TREND"
                peak_val = val
            else:
                current_state = "🔴 STRONG BEARISH TREND (Rally Stopped)"

        states.append(current_state)

    df["Flip_Status"] = states
    return df


def calculate_dynamic_hints(df_in):
    df = df_in.copy()
    hints = []
    for h_diff in df["HAM_Diff_Kalman"]:
        if h_diff > 1.0:
            hints.append("🔥 Strong Bullish Expansion")
        elif h_diff < -1.0:
            hints.append("❄️ Bearish Expansion")
        elif abs(h_diff) <= 0.2:
            hints.append("🎯 Equilibrium Zone")
        else:
            hints.append("⚖️ Neutral Momentum")

    df["HAM_Hint"] = hints
    return df


# =====================================================================
# DATA FETCH ENGINE: 2-YEAR HISTORICAL DAILY MCX DATA
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_2year_mcx_silver():
    tickers = ["SI=F", "SILVERMIC.MCX"]
    data = pd.DataFrame()

    for ticker in tickers:
        try:
            # 2Y Daily interval gives accurate full 2-year horizon (~500 candles)
            data = yf.download(
                ticker, period="2y", interval="1d", progress=False
            )
            if not data.empty and len(data) > 200:
                break
        except Exception:
            continue

    if data.empty:
        raise ValueError("MCX Silver Data Fetch Failed from Yahoo Finance APIs.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df_raw = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df_raw


# Fetch Data
try:
    with st.spinner("🔄 Fetching 2-Year MCX Silver Data & Processing..."):
        df = fetch_2year_mcx_silver()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# FULL CAUSAL KINEMATICS ENGINE (ENTIRE 2-YEAR VECTOR)
# =====================================================================
df = apply_heikin_ashi(df)

# Normal Close Processing
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_causal(
    normal_close_full, window=30
)
kalman_base_normal = apply_kalman_filter_causal(
    normal_close_full, initial_p=50.0, q_val=0.005, r_val=0.2
)
momentum_normal = apply_kalman_filter_causal(
    normal_close_full - kalman_base_normal,
    initial_p=0.50,
    q_val=0.005,
    r_val=0.1,
)
df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

# Hubble & Gaussian Engine
H0_const = 70.0
df["HAM_Expansion_a"] = df["HAM_Normal"]
raw_hubble_vel = H0_const * df["HAM_Expansion_a"].to_numpy()

df["HAM_Hubble_Vel_v"] = apply_bipolar_wave_gaussian_causal(
    raw_hubble_vel, sigma=gaussian_sigma, window_size=25
)
df["HAM_Hubble_Vel_Kalman"] = apply_kalman_filter_causal(
    df["HAM_Hubble_Vel_v"].to_numpy(), initial_p=0.50, q_val=0.005, r_val=0.1
)

# HA Path
ha_close_full = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_causal(ha_close_full, window=30)
kalman_base_ha = apply_kalman_filter_causal(
    ha_close_full, initial_p=50.0, q_val=0.005, r_val=0.2
)
momentum_ha = apply_kalman_filter_causal(
    ha_close_full - kalman_base_ha, initial_p=0.50, q_val=0.005, r_val=0.1
)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

# HAM Diff & Filters
df["HAM_Diff_Raw"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]
df["HAM_Diff_Kalman"] = apply_kalman_filter_causal(
    df["HAM_Diff_Raw"].to_numpy(), initial_p=0.50, q_val=0.005, r_val=0.1
)

df = apply_hysteresis_state_machine(df, reversal_threshold_pct=0.20)
df = calculate_dynamic_hints(df)

# =====================================================================
# 50:50 SPLIT ENGINE (ANALYSIS VS PREDICTION)
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_in_sample = df.iloc[:split_idx].copy()       # First 1 Year (In-Sample Analysis)
df_out_of_sample = df.iloc[split_idx:].copy()  # Second 1 Year (Out-of-Sample Prediction)

st.success(
    f"📊 **Total Dataset:** {total_candles} Daily Candles (~2 Years)\n\n"
    f"🔹 **50% In-Sample Analysis Window:** {len(df_in_sample)} Candles "
    f"({df_in_sample.index[0].strftime('%Y-%m-%d')} to {df_in_sample.index[-1].strftime('%Y-%m-%d')})\n\n"
    f"🔸 **50% Out-Of-Sample Prediction Window:** {len(df_out_of_sample)} Candles "
    f"({df_out_of_sample.index[0].strftime('%Y-%m-%d')} to {df_out_of_sample.index[-1].strftime('%Y-%m-%d')})"
)

# Display Matrix Setup
clean_cols = [
    "Close",
    "HA_Close",
    "Hurst_Normal",
    "HAM_Normal",
    "HAM_Hubble_Vel_v",
    "HAM_Hubble_Vel_Kalman",
    "HAM_HeikinAshi",
    "HAM_Hint",
    "HAM_Diff_Kalman",
    "HAM_Velocity",
    "HAM_Acceleration",
    "Flip_Status",
]

def prepare_display_df(target_df):
    disp = pd.DataFrame(index=target_df.index)
    for col in clean_cols:
        if col not in ["Flip_Status", "HAM_Hint"]:
            disp[col] = np.asarray(target_df[col], dtype=float).flatten()
        else:
            disp[col] = target_df[col]
    disp = disp.iloc[::-1]
    disp.index = disp.index.strftime("%Y-%m-%d IST")
    return disp

tab1, tab2 = st.tabs([
    "🔮 50% Out-of-Sample Prediction Matrix",
    "📊 50% In-Sample Analysis Matrix"
])

with tab1:
    disp_pred = prepare_display_df(df_out_of_sample)
    st.markdown("### 🔸 Out-of-Sample Prediction Horizon")
    st.dataframe(
        disp_pred,
        column_config={
            "Close": st.column_config.NumberColumn("MCX Silver Price", format="%.2f"),
            "HAM_Hubble_Vel_v": st.column_config.NumberColumn(f"🔭 Hubble Wave (+/- Gaussian)", format="%.2f"),
            "HAM_Diff_Kalman": st.column_config.NumberColumn("📊 HAM Diff", format="%.2f"),
            "Flip_Status": st.column_config.TextColumn("🎯 Hysteresis State"),
        },
        use_container_width=True,
        height=500,
    )

with tab2:
    disp_analysis = prepare_display_df(df_in_sample)
    st.markdown("### 🔹 Historical Training & Baseline Analysis Horizon")
    st.dataframe(
        disp_analysis,
        use_container_width=True,
        height=500,
    )
