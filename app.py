from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(page_title="BTC Kinematics & Fokker-Planck Engine", layout="wide")
st.title("⚡ Bitcoin (BTC-USD) Advanced Predictive Engine")
st.write("🎯 **1-Hour Timeframe:** Mean-Reverting Fokker-Planck 24h Target & 95% Volatility Range")

# Sidebar
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=0.50, q_val=0.0001, r_val=0.1
):
    """Standard Causal Kalman Filter Engine."""
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


def calculate_rolling_hurst_vectorized(price_series, window=30):
    arr = np.asarray(price_series, dtype=float).flatten()
    s = pd.Series(arr)
    log_returns = np.log(s / s.shift(1)).fillna(0.0).to_numpy()
    hurst_values = np.full(len(arr), 0.5)

    if len(log_returns) < window:
        return hurst_values

    windows = np.lib.stride_tricks.sliding_window_view(
        log_returns, window_shape=window
    )
    means = np.mean(windows, axis=1, keepdims=True)
    cum_dev = np.cumsum(windows - means, axis=1)

    r_val = np.ptp(cum_dev, axis=1)
    s_val = np.std(windows, axis=1, ddof=1) + 1e-10
    rs_ratio = r_val / s_val

    valid_mask = rs_ratio > 0
    h_calculated = np.full(len(rs_ratio), 0.5)
    h_calculated[valid_mask] = np.log(rs_ratio[valid_mask]) / np.log(window)

    hurst_values[window - 1 : window - 1 + len(h_calculated)] = np.clip(
        h_calculated, 0.0, 1.0
    )
    return hurst_values


# --- THEORY COMPUTATIONS (STRICTLY CAUSAL / NO LOOKAHEAD) ---
def compute_chaos_attractor_dist(price_series, tau=1, window=24):
    """Chaos Theory Metric (Phase Space Attractor Divergence)."""
    s = pd.Series(price_series)
    diff_t = s - s.shift(tau)
    diff_2tau = s.shift(tau) - s.shift(2 * tau)
    
    phase_dist = np.sqrt(diff_t**2 + diff_2tau**2)
    rolling_mean = phase_dist.rolling(window=window).mean()
    rolling_std = phase_dist.rolling(window=window).std() + 1e-8
    chaos_metric = (phase_dist - rolling_mean) / rolling_std
    return chaos_metric.fillna(0.0)


def compute_ramsey_structure_density(price_series, window=24):
    """Ramsey Theory Metric (Guaranteed Micro-pattern Density)."""
    s = pd.Series(price_series)
    returns = s.diff()
    direction = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
    
    pattern_density = pd.Series(direction).rolling(window=window).apply(
        lambda x: np.abs(np.sum(x)) / window, raw=True
    )
    return pattern_density.fillna(0.0)


def compute_takens_trajectory(price_series, lag=1):
    """Takens' Theorem Metric (3D Delay Embedding Velocity)."""
    s = pd.Series(price_series)
    x = s
    y = s.shift(lag)
    z = s.shift(2 * lag)
    
    vx = x.diff()
    vy = y.diff()
    vz = z.diff()
    
    takens_velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    return takens_velocity.fillna(0.0)


# --- MEAN-REVERTING FOKKER-PLANCK (ORNSTEIN-UHLENBECK) ENGINE ---
def compute_fokker_planck_ou_target(price_series, window=30, horizon=24, theta=0.08):
    """
    Solves Mean-Reverting Fokker-Planck Stochastic Differential Equation.
    Calculates Mean-Reverting Expected Target & 95% Confidence Interval Bands for t + 24.
    STRICTLY NO FUTURE LEAKAGE.
    """
    s = pd.Series(price_series, dtype=float)
    log_returns = np.log(s / s.shift(1)).fillna(0.0)
    
    # Anchor Mean & Historical Volatility up to time t
    rolling_mean = s.rolling(window=window).mean()
    rolling_vol = log_returns.rolling(window=window).std() + 1e-8
    
    # 1. Expected Target Mean Price at t + 24
    decay_factor = np.exp(-theta * horizon)
    expected_target_24h = rolling_mean + (s - rolling_mean) * decay_factor
    
    # 2. Cumulative Volatility Diffusion Variance over 24 hours
    diffusion_std = s * rolling_vol * np.sqrt((1 - np.exp(-2 * theta * horizon)) / (2 * theta))
    
    # 3. 95% Confidence Interval Bounds (1.96 * Sigma)
    target_lower_95 = expected_target_24h - (1.96 * diffusion_std)
    target_upper_95 = expected_target_24h + (1.96 * diffusion_std)
    
    return expected_target_24h.fillna(s), target_lower_95.fillna(s), target_upper_95.fillna(s)


def compute_market_regime_state(price_series, window=30):
    """Quantile Volatility & Drift Regime Classification Engine."""
    s = pd.Series(price_series, dtype=float)
    log_returns = np.log(s / s.shift(1)).fillna(0.0)
    
    mu = log_returns.rolling(window=window).mean()
    sigma = log_returns.rolling(window=window).std() + 1e-8
    z_momentum = mu / sigma
    
    regime = np.where(
        z_momentum > 0.35, "Bullish Trend",
        np.where(z_momentum < -0.35, "Bearish Trend", "High Noise / Chop")
    )
    return pd.Series(regime, index=s.index)


# =====================================================================
# DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
    endpoint = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts
    headers = {"User-Agent": "Mozilla/5.0"}

    while current_start < end_ts:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": current_start,
            "limit": 1000,
        }
        res = requests.get(
            endpoint, params=params, headers=headers, timeout=10
        ).json()

        if not isinstance(res, list) or len(res) == 0:
            break

        all_candles.extend(res)
        last_candle_time = res[-1][0]
        if last_candle_time <= current_start:
            break
        current_start = last_candle_time + 1
        time.sleep(0.02)

    if len(all_candles) < 2000:
        return None

    cols = [
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ]
    df_raw = pd.DataFrame(all_candles, columns=cols)
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_raw[num_cols] = df_raw[num_cols].astype(float)
    df_raw["Timestamp"] = pd.to_datetime(
        df_raw["OpenTime"], unit="ms", utc=True
    )
    df_raw.set_index("Timestamp", inplace=True)
    return df_raw[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=3600)
def fetch_coinbase_data(start_dt, now_dt):
    endpoint = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    headers = {"User-Agent": "Mozilla/5.0"}
    current_end = now_dt
    all_candles = []

    while current_end > start_dt:
        current_start = max(start_dt, current_end - timedelta(hours=300))
        params = {
            "granularity": 3600,
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        }
        res = requests.get(
            endpoint, params=params, headers=headers, timeout=10
        ).json()

        if isinstance(res, list) and len(res) > 0:
            all_candles.extend(res)
        else:
            break

        current_end = current_start
        time.sleep(0.05)

    if len(all_candles) == 0:
        return None

    cols = ["time", "Low", "High", "Open", "Close", "Volume"]
    df_raw = pd.DataFrame(all_candles, columns=cols)
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_raw[num_cols] = df_raw[num_cols].astype(float)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["time"], unit="s", utc=True)
    df_raw.set_index("Timestamp", inplace=True)
    df_raw.sort_index(ascending=True, inplace=True)
    return df_raw[["Open", "High", "Low", "Close", "Volume"]]


def get_robust_2year_hourly():
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=730)

    try:
        df = fetch_binance_data(
            int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
        )
        if df is not None and len(df) >= 5000:
            return df, "Binance REST API"
    except Exception:
        pass

    df = fetch_coinbase_data(start_dt, now)
    if df is not None and len(df) >= 2000:
        return df, "Coinbase Pro API (Fallback)"

    raise ValueError("Failed to fetch price data.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching Data & Calculating Engine..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# CALCULATION ENGINE
# =====================================================================
# 1. Base HAM Normal Signal
normal_close_full = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close_full, window=30
)
kalman_base_normal = apply_kalman_filter_custom(
    normal_close_full, initial_p=50.0, q_val=0.0005, r_val=0.2
)
momentum_normal = apply_kalman_filter_custom(
    normal_close_full - kalman_base_normal,
    initial_p=0.50,
    q_val=0.001,
    r_val=0.1,
)
df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

# 2. Chaos Z-Score & 0.50 Kalman Smoothing
df["Chaos_Attractor_ZScore"] = compute_chaos_attractor_dist(df["Close"])
chaos_raw = df["Chaos_Attractor_ZScore"].to_numpy()
df["Chaos_ZScore_Kalman"] = apply_kalman_filter_custom(
    chaos_raw, initial_p=0.50, q_val=0.001, r_val=0.1
)

# 3. Auxiliary Theory Columns
df["Ramsey_Order_Density"] = compute_ramsey_structure_density(df["Close"])
df["Takens_Embedding_Velocity"] = compute_takens_trajectory(df["Close"])

# 4. PREDICTIVE MEAN-REVERTING FOKKER-PLANCK ENGINE
target_24h, lower_95, upper_95 = compute_fokker_planck_ou_target(df["Close"], window=30, horizon=24)
df["Predicted_Close_24h"] = target_24h
df["Target_Lower_95%"] = lower_95
df["Target_Upper_95%"] = upper_95
df["Market_Regime_State"] = compute_market_regime_state(df["Close"], window=30)


# =====================================================================
# DISPLAY MATRIX & METRICS
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)
df_predict = df.iloc[split_idx:].copy()

clean_cols = [
    "Close", 
    "Predicted_Close_24h",
    "Target_Lower_95%",
    "Target_Upper_95%",
    "Market_Regime_State",
    "HAM_Normal", 
    "Chaos_Attractor_ZScore",
    "Chaos_ZScore_Kalman",
    "Ramsey_Order_Density",
    "Takens_Embedding_Velocity"
]

display_df = pd.DataFrame(index=df_predict.index)
for col in clean_cols:
    display_df[col] = df_predict[col].values

# Display latest candles at the top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

# Calculate Delta Target
curr_close = float(latest_candle['Close'])
target_24h = float(latest_candle['Predicted_Close_24h'])
delta_val = target_24h - curr_close
delta_pct = (delta_val / curr_close) * 100.0

# Metrics Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${curr_close:,.2f}")
col2.metric(
    "Fokker-Planck 24h Target", 
    f"${target_24h:,.2f}", 
    f"{delta_val:+,.2f} ({delta_pct:+.2f}%)"
)
col3.metric(
    "95% Probability Range", 
    f"${float(latest_candle['Target_Lower_95\%']):,.2f} -${float(latest_candle['Target_Upper_95%']):,.2f}"
)
col4.metric("Current Market Regime", f"{latest_candle['Market_Regime_State']}")

st.divider()

# Interactive Data Frame
st.subheader(f"📋 Corrected Predictive Kinematic Matrix ({len(display_df):,} Locked Candles)")
st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price ($)", format="$%.2f"
        ),
        "Predicted_Close_24h": st.column_config.NumberColumn(
            "Fokker-Planck 24h Target ($)", format="$%.2f"
        ),
        "Target_Lower_95%": st.column_config.NumberColumn(
            "Lower 95% Limit ($)", format="$%.2f"
        ),
        "Target_Upper_95%": st.column_config.NumberColumn(
            "Upper 95% Limit ($)", format="$%.2f"
        ),
        "Market_Regime_State": st.column_config.TextColumn(
            "Regime State"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "HAM Normal (Kalman)", format="%.4f"
        ),
        "Chaos_Attractor_ZScore": st.column_config.NumberColumn(
            "Chaos Raw (Z-Score)", format="%.4f"
        ),
        "Chaos_ZScore_Kalman": st.column_config.NumberColumn(
            "Chaos Smoothed (0.50 Kalman)", format="%.4f"
        ),
        "Ramsey_Order_Density": st.column_config.NumberColumn(
            "Ramsey Structure Density", format="%.4f"
        ),
        "Takens_Embedding_Velocity": st.column_config.NumberColumn(
            "Takens Phase Velocity", format="%.4f"
        ),
    },
    use_container_width=True,
    height=600,
)
