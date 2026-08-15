import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC 3000-Tree Kinematic ML Engine", layout="wide"
)
st.title("⚡ Bitcoin (BTC-USD) 3,000-Tree Multi-Point Kinematic Engine")
st.write(
    "🎯 **1-Hour Timeframe Engine:** 3,000 Decision Trees | Multi-Point Feature Array | Strict Consensus Thresholds | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🤖 **ML Model:** 3,000 Decision Trees\n\n🎯 **Multi-Point Array:** 12 Kinematic Features"
)


# =====================================================================
# MATHEMATICAL & MULTI-POINT FEATURE ENGINES
# =====================================================================
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


def compute_multipoint_features(df_in):
    df = df_in.copy()

    # 1. Base HAM & HA Calculations
    df["Base_HAM_Normal"] = df["Close"].diff(14).ewm(span=9).mean().fillna(0.0)
    df["HAM_HA_Signal"] = df["HA_Close"].diff(14).ewm(span=14).mean().fillna(0.0)
    df["HAM_Diff"] = df["Base_HAM_Normal"] - df["HAM_HA_Signal"]

    # 2. Higher-Order Kinematics
    df["HAM_Velocity"] = df["HAM_Diff"].diff().fillna(0.0)
    df["HAM_Acceleration"] = df["HAM_Velocity"].diff().fillna(0.0)
    df["HAM_Jerk"] = df["HAM_Acceleration"].diff().fillna(0.0)  # 3rd Derivative

    # 3. Multi-Period Return Vectors
    df["Ret_1H"] = df["Close"].pct_change(1).fillna(0.0)
    df["Ret_3H"] = df["Close"].pct_change(3).fillna(0.0)
    df["Ret_6H"] = df["Close"].pct_change(6).fillna(0.0)
    df["Ret_12H"] = df["Close"].pct_change(12).fillna(0.0)

    # 4. Volatility & Candle Dynamics
    high_low_range = df["High"] - df["Low"]
    df["Vol_ATR_Ratio"] = (high_low_range / df["Close"]).fillna(0.0)
    df["Vol_SMA_Ratio"] = (df["Volume"] / df["Volume"].rolling(20).mean()).fillna(1.0)

    # 5. Target Variable (Direction of Next Candle)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    return df


# =====================================================================
# 3,000 DECISION TREES MACHINE LEARNING ENGINE
# =====================================================================
def train_and_predict_3000_trees(df_in):
    df = df_in.copy()

    # Define 12 Multi-Point Features
    features = [
        "Base_HAM_Normal",
        "HAM_HA_Signal",
        "HAM_Diff",
        "HAM_Velocity",
        "HAM_Acceleration",
        "HAM_Jerk",
        "Ret_1H",
        "Ret_3H",
        "Ret_6H",
        "Ret_12H",
        "Vol_ATR_Ratio",
        "Vol_SMA_Ratio",
    ]

    df.dropna(subset=features, inplace=True)

    # 50:50 Train:Predict Split
    total_candles = len(df)
    split_idx = int(total_candles * 0.50)

    df_train = df.iloc[:split_idx].copy()
    df_predict = df.iloc[split_idx:].copy()

    X_train = df_train[features]
    y_train = df_train["Target"]

    X_predict = df_predict[features]

    # Initialize 3,000 Decision Trees Random Forest Model
    model = RandomForestClassifier(
        n_estimators=3000,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    # Fit Model on 50% Backdata
    model.fit(X_train, y_train)

    # Calculate Probability Votes from 3,000 Trees
    probs = model.predict_proba(X_predict)
    bullish_vote_ratio = probs[:, 1]
    bearish_vote_ratio = probs[:, 0]

    # Strict Probability Threshold Filtering
    signals = []
    for bull_p, bear_p in zip(bullish_vote_ratio, bearish_vote_ratio):
        if bull_p >= 0.70:  # Strong Bullish Consensus
            signals.append("🟢 HIGH CONVICTION BUY (3,000 Trees)")
        elif bear_p >= 0.70:  # Strong Bearish Consensus
            signals.append("🔴 HIGH CONVICTION SELL (3,000 Trees)")
        else:
            signals.append("🟡 NO TRADE / RANGE NOISE (Divided Trees)")

    df_predict["Bullish_Vote_Pct"] = (bullish_vote_ratio * 100).round(1)
    df_predict["ML_3000_Signal"] = signals

    return df_train, df_predict, model


# =====================================================================
# DUAL-SOURCE DATA FETCH ENGINE
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

    raise ValueError("Failed to fetch historical data from primary and fallback sources.")


# Fetch Data & Train Model
try:
    with st.spinner("🔄 Training 3,000 Decision Trees across Multi-Point Array..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]  # Drop unclosed running candle
        df.index = df.index.tz_convert("Asia/Kolkata")

        # Calculations & Features
        df = apply_heikin_ashi(df)
        df = compute_multipoint_features(df)
        df_train, df_predict, trained_model = train_and_predict_3000_trees(df)

except Exception as e:
    st.error(f"🚨 Engine Error: {e}")
    st.stop()


# =====================================================================
# DISPLAY MATRIX (IST)
# =====================================================================
clean_cols = [
    "Close",
    "Base_HAM_Normal",
    "HAM_HA_Signal",
    "HAM_Diff",
    "HAM_Velocity",
    "HAM_Acceleration",
    "HAM_Jerk",
    "Vol_ATR_Ratio",
    "Bullish_Vote_Pct",
    "ML_3000_Signal",
]

display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    if col != "ML_3000_Signal":
        display_df[col] = (
            np.asarray(df_predict[col], dtype=float).flatten().round(2)
        )
    else:
        display_df[col] = df_predict[col]

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Base HAM Normal", f"{latest_candle['Base_HAM_Normal']:.2f}")
col3.metric("HAM Diff", f"{latest_candle['HAM_Diff']:.2f}")
col4.metric(
    "🌳 3,000-Tree Bull Vote", f"{latest_candle['Bullish_Vote_Pct']}%"
)
col5.metric("🎯 ML Consensus Signal", f"{latest_candle['ML_3000_Signal']}")

st.divider()

st.subheader(
    f"📋 3,000-Tree Multi-Point Kinematic Matrix ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "Base_HAM_Normal": st.column_config.NumberColumn("Base HAM Normal", format="%.2f"),
        "HAM_HA_Signal": st.column_config.NumberColumn("HAM HA Signal", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("📊 HAM Diff", format="%.2f"),
        "HAM_Velocity": st.column_config.NumberColumn("⚡ Velocity (Δ1)", format="%.2f"),
        "HAM_Acceleration": st.column_config.NumberColumn("🚀 Accel (Δ2)", format="%.2f"),
        "HAM_Jerk": st.column_config.NumberColumn("💥 Jerk (Δ3)", format="%.2f"),
        "Vol_ATR_Ratio": st.column_config.NumberColumn("🌊 ATR Ratio", format="%.4f"),
        "Bullish_Vote_Pct": st.column_config.NumberColumn("🌳 Bull Vote (%)", format="%.1f%%"),
        "ML_3000_Signal": st.column_config.TextColumn("🎯 3,000-Tree Consensus Signal"),
    },
    use_container_width=True,
    height=600,
)
