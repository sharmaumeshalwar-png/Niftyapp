import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="BTC Kinematics & SL Gap Engine", layout="wide"
)
st.title("⚡ BTC Dynamic TSL & SL Gap Engine")
st.write(
    "🎯 **1-Hour Liquidity Engine:** Tracking Dynamic TSL, Long/Short SL"
    " Levels & Candle-to-Candle SL Shift Gaps"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()


# =====================================================================
# MATHEMATICAL ENGINES
# =====================================================================
def apply_kalman_filter_custom(
    data_array, initial_p=50.0, q_val=0.0001, r_val=0.001
):
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


# =====================================================================
# MULTI-ENDPOINT DATA FETCHING
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_binance_data_robust(start_ts, end_ts):
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }

    for endpoint in endpoints:
        all_candles = []
        current_start = start_ts
        try:
            while current_start < end_ts:
                params = {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "startTime": current_start,
                    "limit": 1000,
                }
                res = requests.get(
                    endpoint, params=params, headers=headers, timeout=5
                )

                if res.status_code != 200:
                    break

                data = res.json()
                if not isinstance(data, list) or len(data) == 0:
                    break

                all_candles.extend(data)
                last_candle_time = data[-1][0]
                if last_candle_time <= current_start:
                    break
                current_start = last_candle_time + 1
                time.sleep(0.01)

            if len(all_candles) >= 200:
                cols = [
                    "OpenTime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "CloseTime",
                    "QuoteVolume",
                    "Trades",
                    "TakerBase",
                    "TakerQuote",
                    "Ignore",
                ]
                df_raw = pd.DataFrame(all_candles, columns=cols)
                num_cols = ["Open", "High", "Low", "Close", "Volume"]
                df_raw[num_cols] = df_raw[num_cols].astype(float)
                df_raw["Timestamp"] = pd.to_datetime(
                    df_raw["OpenTime"], unit="ms", utc=True
                )
                df_raw.set_index("Timestamp", inplace=True)
                return df_raw[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue

    return pd.DataFrame()


# --- FETCH DATA ---
try:
    with st.spinner("🔄 Fetching Market Data & Calculating SL Gaps..."):
        now = datetime.now(timezone.utc)
        start_dt = now - timedelta(days=180)
        df = fetch_binance_data_robust(
            int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000)
        )

        if df.empty:
            st.error(
                "❌ Data fetch error. Click 'Force Refresh Engine' or turn on VPN."
            )
            st.stop()

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")].iloc[:-1]
        df.index = df.index.tz_convert("Asia/Kolkata")
except Exception as e:
    st.error(f"Data Engine Error: {e}")
    st.stop()

# =====================================================================
# KINEMATICS & DYNAMIC STOP LOSS ENGINE
# =====================================================================
df = apply_heikin_ashi(df)

# Hurst & Momentum
normal_close = np.asarray(df["Close"], dtype=float).flatten()
df["Hurst_Normal"] = calculate_rolling_hurst_vectorized(
    normal_close, window=30
)
kalman_base_normal = apply_kalman_filter_custom(
    normal_close, initial_p=50.0, q_val=0.0001, r_val=0.001
)
momentum_normal = apply_kalman_filter_custom(
    normal_close - kalman_base_normal,
    initial_p=0.50,
    q_val=0.0001,
    r_val=0.001,
)
df["HAM_Normal"] = momentum_normal * (df["Hurst_Normal"].to_numpy() * 2.0)

ha_close = np.asarray(df["HA_Close"], dtype=float).flatten()
df["Hurst_HA"] = calculate_rolling_hurst_vectorized(ha_close, window=30)
kalman_base_ha = apply_kalman_filter_custom(
    ha_close, initial_p=50.0, q_val=0.0001, r_val=0.001
)
momentum_ha = apply_kalman_filter_custom(
    ha_close - kalman_base_ha, initial_p=0.50, q_val=0.0001, r_val=0.001
)
df["HAM_HeikinAshi"] = momentum_ha * (df["Hurst_HA"].to_numpy() * 2.0)

df["HAM_Diff"] = df["HAM_Normal"] - df["HAM_HeikinAshi"]

# --- ATR & DYNAMIC STOP LOSS LEVELS ---
tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - df["Close"].shift(1)).abs()
tr3 = (df["Low"] - df["Close"].shift(1)).abs()
df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

df["Dynamic_Short_SL"] = df["High"] + (
    df["ATR"] * (1.5 - df["Hurst_Normal"] * 0.5)
)
df["Dynamic_Long_SL"] = df["Low"] - (
    df["ATR"] * (1.5 - df["Hurst_Normal"] * 0.5)
)

# =====================================================================
# NEW: CALCULATE SL UP / DOWN GAPS
# =====================================================================
df["Long_SL_Gap"] = (
    df["Dynamic_Long_SL"] - df["Dynamic_Long_SL"].shift(1)
).fillna(0.0)
df["Short_SL_Gap"] = (
    df["Dynamic_Short_SL"] - df["Dynamic_Short_SL"].shift(1)
).fillna(0.0)


# SL Behavior
def analyze_sl_behavior(row):
    diff = row["HAM_Diff"]
    high = row["High"]
    low = row["Low"]
    long_sl = row["Dynamic_Long_SL"]
    short_sl = row["Dynamic_Short_SL"]

    if diff < -15 and low <= long_sl:
        return "⚡ Long SL Swept! (Bullish Reversal Zone)"
    elif diff > 15 and high >= short_sl:
        return "🚨 Short SL Swept! (Bearish Reversal Zone)"
    elif diff > 5:
        return "📈 Bullish Momentum (Longs Safe)"
    elif diff < -5:
        return "📉 Bearish Momentum (Shorts Safe)"
    else:
        return "⚖️ Neutral / Liquidity Building"


df["SL_Behavior"] = df.apply(analyze_sl_behavior, axis=1)

# =====================================================================
# TSL (TRAILING STOP LOSS) COLUMN
# =====================================================================
abs_ham = df["HAM_Normal"].abs()
df["SL_Distance"] = np.where(
    abs_ham > 20,
    df["ATR"] * 0.8,
    np.where(abs_ham > 5, df["ATR"] * 1.2, df["ATR"] * 1.8),
)

closes = df["Close"].to_numpy()
highs = df["High"].to_numpy()
lows = df["Low"].to_numpy()
distances = df["SL_Distance"].fillna(0).to_numpy()
ham_vals = df["HAM_Normal"].to_numpy()

tsl_arr = np.zeros(len(df))
curr_long_sl = lows[0] - distances[0]
curr_short_sl = highs[0] + distances[0]

for i in range(1, len(df)):
    raw_long = lows[i] - distances[i]
    raw_short = highs[i] + distances[i]

    curr_long_sl = max(curr_long_sl, raw_long)
    curr_short_sl = min(curr_short_sl, raw_short)

    if closes[i] < curr_long_sl:
        curr_long_sl = lows[i] - distances[i]
    if closes[i] > curr_short_sl:
        curr_short_sl = highs[i] + distances[i]

    if ham_vals[i] >= 0:
        tsl_arr[i] = curr_long_sl
    else:
        tsl_arr[i] = curr_short_sl

df["TSL"] = tsl_arr

# =====================================================================
# DISPLAY TABLE
# =====================================================================
split_idx = int(len(df) * 0.50)
df_predict = df.iloc[split_idx:].copy().dropna()

clean_cols = [
    "Close",
    "TSL",
    "Dynamic_Long_SL",
    "Long_SL_Gap",  # Added Gap Column
    "Dynamic_Short_SL",
    "Short_SL_Gap",  # Added Gap Column
    "HAM_Normal",
    "HAM_HeikinAshi",
    "HAM_Diff",
    "SL_Behavior",
]
display_df = df_predict[clean_cols].copy().iloc[::-1]

display_df["Close"] = display_df["Close"].round(2)
display_df["TSL"] = display_df["TSL"].round(2)
display_df["Dynamic_Long_SL"] = display_df["Dynamic_Long_SL"].round(2)
display_df["Long_SL_Gap"] = display_df["Long_SL_Gap"].round(2)
display_df["Dynamic_Short_SL"] = display_df["Dynamic_Short_SL"].round(2)
display_df["Short_SL_Gap"] = display_df["Short_SL_Gap"].round(2)
display_df["HAM_Normal"] = display_df["HAM_Normal"].round(2)
display_df["HAM_HeikinAshi"] = display_df["HAM_HeikinAshi"].round(2)
display_df["HAM_Diff"] = display_df["HAM_Diff"].round(2)
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest = display_df.iloc[0]
st.markdown(f"### 🔒 **LAST CANDLE STATUS (IST):** `{display_df.index[0]}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Close", f"${latest['Close']:,.2f}")
c2.metric("Active TSL", f"${latest['TSL']:,.2f}")
c3.metric(
    "Long SL Gap",
    f"${latest['Long_SL_Gap']:,.2f}",
    delta=f"{latest['Long_SL_Gap']:.2f} pts",
)
c4.metric(
    "Short SL Gap",
    f"${latest['Short_SL_Gap']:,.2f}",
    delta=f"{latest['Short_SL_Gap']:.2f} pts",
)

st.divider()
st.subheader("📋 Candle-by-Candle Stop Loss Shift & Liquidity Matrix")

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close ($)", format="$%.2f"),
        "TSL": st.column_config.NumberColumn("TSL ($)", format="$%.2f"),
        "Dynamic_Long_SL": st.column_config.NumberColumn(
            "Long SL Level ($)", format="$%.2f"
        ),
        "Long_SL_Gap": st.column_config.NumberColumn(
            "Long SL Shift Gap ($)", format="$%.2f"
        ),
        "Dynamic_Short_SL": st.column_config.NumberColumn(
            "Short SL Level ($)", format="$%.2f"
        ),
        "Short_SL_Gap": st.column_config.NumberColumn(
            "Short SL Shift Gap ($)", format="$%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "HAM Normal", format="%.2f"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA", format="%.2f"
        ),
        "HAM_Diff": st.column_config.NumberColumn("HAM Diff", format="%.2f"),
        "SL_Behavior": st.column_config.TextColumn("Har Candle Ka Behavior"),
    },
    use_container_width=True,
    height=600,
)
