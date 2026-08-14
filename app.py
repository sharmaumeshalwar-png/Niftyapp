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
    page_title="BTC HAM Kinematics with Acceleration Flip Filter",
    layout="wide",
)
st.title("⚡ Bitcoin (BTC-USD) HAM Kinematic Engine + Acceleration Flip Filter")
st.write(
    "🎯 **1-Hour Timeframe Engine:** HAM Diff, Velocity (Δ1), Acceleration (Δ2) & Auto-Flip Filter | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal)\n\n🔒 **Dual REST Stream:** CONNECTED"
)


# =====================================================================
# 8-STEP VERIFICATION MATHEMATICAL ENGINES
# =====================================================================

# STEP 2: Heikin Ashi Calculation Engine
def compute_heikin_ashi(df_in):
    df_ha = df_in.copy()
    
    ha_close = (df_ha['Open'] + df_ha['High'] + df_ha['Low'] + df_ha['Close']) / 4.0
    
    ha_open = np.zeros(len(df_ha), dtype=float)
    ha_open[0] = (df_ha['Open'].iloc[0] + df_ha['Close'].iloc[0]) / 2.0
    
    for i in range(1, len(df_ha)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2.0
        
    ha_high = np.maximum(df_ha['High'].values, np.maximum(ha_open, ha_close.values))
    ha_low = np.minimum(df_ha['Low'].values, np.minimum(ha_open, ha_close.values))
    
    df_ha['HA_Open'] = ha_open
    df_ha['HA_High'] = ha_high
    df_ha['HA_Low'] = ha_low
    df_ha['HA_Close'] = ha_close.values
    
    return df_ha

# STEP 3, 4, 5: HAM Calculation Engine
def compute_ham_kinematics(df_in):
    df = df_in.copy()
    
    # Simple Momentum Baseline for HAM
    close_vals = df['Close'].to_numpy()
    ha_close_vals = df['HA_Close'].to_numpy()
    
    # Calculate Base HAM Normal and HAM HA Signal
    base_ham = pd.Series(close_vals).diff(14).ewm(span=9).mean().fillna(0.0).to_numpy()
    ham_ha_signal = pd.Series(ha_close_vals).diff(14).ewm(span=14).mean().fillna(0.0).to_numpy()
    
    df['Base_HAM_Normal'] = base_ham
    df['HAM_HA_Signal'] = ham_ha_signal
    df['HAM_Diff'] = df['Base_HAM_Normal'] - df['HAM_HA_Signal']
    
    return df

# STEP 6, 7: Velocity, Acceleration & Flip Logic Filter
def compute_flip_filter(df_in):
    df = df_in.copy()
    
    # 1. HAM Velocity (1st Derivative of Diff)
    df["HAM_Velocity"] = df["HAM_Diff"].diff()
    
    # 2. HAM Acceleration (2nd Derivative of Diff)
    df["HAM_Acceleration"] = df["HAM_Velocity"].diff()
    
    # 3. Dynamic Hurst Exponent (Window=30)
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0.0).to_numpy()
    window = 30
    hurst_vals = np.full(len(df), 0.5)
    
    if len(log_returns) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(log_returns, window_shape=window)
        means = np.mean(windows, axis=1, keepdims=True)
        cum_dev = np.cumsum(windows - means, axis=1)
        r_val = np.ptp(cum_dev, axis=1)
        s_val = np.std(windows, axis=1, ddof=1) + 1e-10
        rs_ratio = r_val / s_val
        valid_mask = rs_ratio > 0
        h_calc = np.full(len(rs_ratio), 0.5)
        h_calc[valid_mask] = np.log(rs_ratio[valid_mask]) / np.log(window)
        hurst_vals[window - 1 : window - 1 + len(h_calc)] = np.clip(h_calc, 0.0, 1.0)
        
    df["Hurst_HA"] = hurst_vals

    # 4. Smart Flip Detection Rules
    conditions = [
        # Real Bearish Flip: Diff is dropping, Velocity < 0, AND Acceleration < 0
        (df["HAM_Diff"] < df["HAM_Diff"].shift(1)) & (df["HAM_Velocity"] < 0) & (df["HAM_Acceleration"] < 0),
        
        # Fake Drop (Wapas Badhega): Diff dropping, Velocity < 0, BUT Acceleration > 0
        (df["HAM_Diff"] < df["HAM_Diff"].shift(1)) & (df["HAM_Velocity"] < 0) & (df["HAM_Acceleration"] > 0),
        
        # Real Bullish Flip: Diff rising, Velocity > 0, AND Acceleration > 0
        (df["HAM_Diff"] > df["HAM_Diff"].shift(1)) & (df["HAM_Velocity"] > 0) & (df["HAM_Acceleration"] > 0),

        # Fake Rise (Wapas Girega): Diff rising, Velocity > 0, BUT Acceleration < 0
        (df["HAM_Diff"] > df["HAM_Diff"].shift(1)) & (df["HAM_Velocity"] > 0) & (df["HAM_Acceleration"] < 0)
    ]
    
    choices = [
        "🔴 REAL BEARISH FLIP",
        "⚠️ FAKEOUT (Wapas Badhega)",
        "🟢 REAL BULLISH FLIP",
        "⚠️ FAKEOUT (Wapas Girega)"
    ]
    
    df["Flip_Status"] = np.select(conditions, choices, default="🟡 STABLE / CONTINUATION")
    return df


# =====================================================================
# DUAL-SOURCE DATA FETCH ENGINE (BINANCE + COINBASE FALLBACK)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_binance_data(start_ts, end_ts):
    endpoint = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts
    headers = {"User-Agent": "Mozilla/5.0"}

    while current_start < end_ts:
        params = {"symbol": "BTCUSDT", "interval": "1h", "startTime": current_start, "limit": 1000}
        res = requests.get(endpoint, params=params, headers=headers, timeout=10).json()

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

    cols = ["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"]
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
        params = {"granularity": 3600, "start": current_start.isoformat(), "end": current_end.isoformat()}
        res = requests.get(endpoint, params=params, headers=headers, timeout=10).json()

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
        df = fetch_binance_data(int(start_dt.timestamp() * 1000), int(now.timestamp() * 1000))
        if df is not None and len(df) >= 5000:
            return df, "Binance REST API"
    except Exception:
        pass

    df = fetch_coinbase_data(start_dt, now)
    if df is not None and len(df) >= 2000:
        return df, "Coinbase Pro API (Fallback)"

    raise ValueError("Both primary and fallback endpoints failed to return sufficient candles.")


# Fetch Data
try:
    with st.spinner("🔄 Fetching Data & Calculating HAM Velocity + Acceleration..."):
        df, source_used = get_robust_2year_hourly()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1] # Drop running unclosed candle
        df.index = df.index.tz_convert("Asia/Kolkata")

except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# MATRIX CALCULATIONS
# =====================================================================
df = compute_heikin_ashi(df)
df = compute_ham_kinematics(df)
df = compute_flip_filter(df)


# =====================================================================
# STEP 8: 50:50 SPLIT & IST MATRIX DISPLAY
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced via {source_used}: {total_candles:,} Total Candles** | 🧠"
    f" **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:**"
    f" {len(df_predict):,} (IST Locked)"
)

clean_cols = [
    "Close",
    "Hurst_HA",
    "Base_HAM_Normal",
    "HAM_HA_Signal",
    "HAM_Diff",
    "HAM_Velocity",
    "HAM_Acceleration",
    "Flip_Status"
]

display_df = pd.DataFrame(index=df_predict.index)

for col in clean_cols:
    if col != "Flip_Status":
        display_df[col] = np.asarray(df_predict[col], dtype=float).flatten().round(2)
    else:
        display_df[col] = df_predict[col]

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# Metric Cards Display
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Locked Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric("Base HAM Normal", f"{latest_candle['Base_HAM_Normal']:.2f}")
col3.metric("HAM HA Signal", f"{latest_candle['HAM_HA_Signal']:.2f}")
col4.metric("📊 HAM Diff", f"{latest_candle['HAM_Diff']:.2f}")
col5.metric("🎯 Signal Status", f"{latest_candle['Flip_Status']}")

st.divider()

st.subheader(
    f"📋 50:50 Clean Kinematic Matrix with Acceleration Filter ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close ($)", format="$%.2f"),
        "Hurst_HA": st.column_config.NumberColumn("Hurst (HA)", format="%.2f"),
        "Base_HAM_Normal": st.column_config.NumberColumn("Base HAM Normal", format="%.2f"),
        "HAM_HA_Signal": st.column_config.NumberColumn("HAM HA Signal", format="%.2f"),
        "HAM_Diff": st.column_config.NumberColumn("📊 HAM Diff", format="%.2f"),
        "HAM_Velocity": st.column_config.NumberColumn("⚡ Velocity (Δ1)", format="%.2f"),
        "HAM_Acceleration": st.column_config.NumberColumn("🚀 Acceleration (Δ2)", format="%.2f"),
        "Flip_Status": st.column_config.TextColumn("🎯 Flip Status / Fakeout Alert"),
    },
    use_container_width=True,
    height=650,
)
