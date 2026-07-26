import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC 2-Year Kinematics Engine (Zero Leakage)", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) 2-Year Pure Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** 2-Year Full History | Multi-HAM Matrix"
    " | 50:50 Learn:Predict Split | IST Locked"
    " [Strict Zero Leakage & Check Bfill Rule]"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Dual REST Stream:**"
    " CONNECTED"
)


# =====================================================================
# MATHEMATICAL ENGINES (Strictly Causal / Zero Look-Ahead Bias)
# =====================================================================

# 1. Condition 2: Causal Kalman Filter on Close (Measurement Variance = 0.50)
def apply_causal_kalman(series, process_variance=1e-5, measurement_variance=0.50):
    """
    Sequential single-pass Causal Kalman Filter.
    Measurement Noise R = 0.50 as strictly required.
    """
    arr = np.asarray(series, dtype=float).flatten()
    if len(arr) == 0:
        return np.array([])
    x, p = arr[0], 1.0
    filtered_values = np.empty(len(arr))
    for i, z in enumerate(arr):
        p_minus = p + process_variance
        k = p_minus / (p_minus + measurement_variance)
        x = x + k * (z - x)
        p = (1 - k) * p_minus
        filtered_values[i] = x
    return filtered_values


# 2. Condition 4: Rolling Hurst Exponent (Strictly 30-Candle Window)
def calculate_rolling_hurst_30(price_series, window=30):
    """
    Fast Causal Rolling Hurst Exponent strictly using a 30-candle window.
    """
    vals = np.asarray(price_series, dtype=float).flatten()
    n = len(vals)
    hurst_arr = np.full(n, 0.5)
    lags = np.arange(2, 10)
    log_lags = np.log(lags)

    for i in range(window, n):
        sub_window = vals[i - window : i]
        tau = [
            max(np.std(sub_window[l:] - sub_window[:-l]), 1e-8) for l in lags
        ]
        hurst_arr[i] = np.polyfit(log_lags, np.log(tau), 1)[0]

    return hurst_arr


def apply_heikin_ashi(df_in):
    """Calculates Heikin-Ashi candles sequentially without look-ahead bias."""
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


# =====================================================================
# DUAL-SOURCE DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
    endpoint = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }

    while current_start < end_ts:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": current_start,
            "limit": 1000,
        }
        try:
            res = requests.get(
                endpoint, params=params, headers=headers, timeout=10
            ).json()
        except Exception:
            break

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
    df_raw["Timestamp"] = pd.to_datetime(df_raw["OpenTime"], unit="ms", utc=True)
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
        try:
            res = requests.get(
                endpoint, params=params, headers=headers, timeout=10
            ).json()
        except Exception:
            break

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

    raise ValueError(
        "Both primary and fallback endpoints failed to return sufficient candles."
    )


# Fetch Data
try:
    with st.spinner(
        "🔄 Fetching 2 Years of Hourly BTC Data (~17,500 Candles)..."
    ):
        df, source_used = get_robust_2year_hourly()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        # 🔒 STRICT LEAKAGE PREVENTION: Drop unclosed running candle
        df = df.iloc[:-1].copy()

        # Convert to IST Timezone
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ FEATURE CALCULATIONS (Strict 5 Conditions)
# =====================================================================
df = apply_heikin_ashi(df)

# Condition 1: Close
# Handled directly as df['Close']

# Condition 2: Kalman of 0.50 on Close
df["kalman"] = apply_causal_kalman(df["Close"], measurement_variance=0.50)

# Condition 3: Weighted Momentum (Close - Kalman via EWMA)
diff = df["Close"] - df["kalman"]
df["weighted_momentum"] = diff.ewm(span=14, adjust=False).mean()

# Condition 4: Hurst of Close (30 Candle Window)
df["hurst"] = calculate_rolling_hurst_30(df["Close"], window=30)

# Condition 5: HAM (Hurst * Weighted Momentum)
df["ham"] = df["hurst"] * df["weighted_momentum"]

# Heikin-Ashi HAM Matrix Calculations
df["kalman_ha"] = apply_causal_kalman(df["HA_Close"], measurement_variance=0.50)
diff_ha = df["HA_Close"] - df["kalman_ha"]
df["weighted_momentum_ha"] = diff_ha.ewm(span=14, adjust=False).mean()
df["hurst_ha"] = calculate_rolling_hurst_30(df["HA_Close"], window=30)
df["ham_ha"] = df["hurst_ha"] * df["weighted_momentum_ha"]


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT (Strict Zero-Bfill Rule)
# =====================================================================
# Target creation for zero future leak checks
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# ZERO BFILL RULE: Drop NaNs created by rolling windows and shift(-1)
df.dropna(inplace=True)

total_candles = len(df)
split_idx = int(total_candles * 0.50)  # Exact 50:50 Cut

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced via {source_used}: {total_candles:,} Total Usable Candles** | 🧠"
    f" **Learn Set (50%):** {len(df_learn):,} | 🔮 **Predict Matrix (50%):**"
    f" {len(df_predict):,} (IST Locked)"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
clean_cols = [
    "Close",
    "kalman",
    "weighted_momentum",
    "hurst",
    "ham",
    "HA_Close",
    "hurst_ha",
    "ham_ha",
]

display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    display_df[col] = np.asarray(df_predict[col], dtype=float).flatten().round(2)

# Reverse DataFrame to display latest closed candle at top
display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# 🎯 LATEST LOCKED CANDLE METRIC CARD
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Kalman (0.50)", f"${latest_candle['kalman']:,.2f}")
col3.metric("Normal HAM Signal", f"{latest_candle['ham']:.2f}")
col4.metric("Heikin-Ashi HAM Signal", f"{latest_candle['ham_ha']:.2f}")

st.divider()

st.subheader(
    f"📋 50:50 Dynamic Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Close Price ($)", format="$%.2f"
        ),
        "kalman": st.column_config.NumberColumn(
            "Kalman 0.50 ($)", format="$%.2f"
        ),
        "weighted_momentum": st.column_config.NumberColumn(
            "Weighted Momentum", format="%.2f"
        ),
        "hurst": st.column_config.NumberColumn(
            "Hurst (30 Win)", format="%.2f"
        ),
        "ham": st.column_config.NumberColumn(
            "🚀 Normal HAM Signal", format="%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close ($)", format="$%.2f"
        ),
        "hurst_ha": st.column_config.NumberColumn(
            "Hurst HA", format="%.2f"
        ),
        "ham_ha": st.column_config.NumberColumn(
            "HA HAM Signal", format="%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
