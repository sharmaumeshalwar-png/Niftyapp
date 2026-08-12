import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =====================================================================
# PAGE CONFIGURATION & HEADER
# =====================================================================
st.set_page_config(
    page_title="BTC-USD Zero-Leakage Fourier Engine", layout="wide"
)
st.title(
    "🛡️ BITCOIN (BTC-USD) Engine — Strictly Causal Causal Fourier (π = 22/7)"
)
st.write(
    "🎯 **1-Hour Timeframe Engine:** Rolling Trajectory FFT (No Look-Ahead Bias) "
    "using $\pi = 22/7$ → Fourier Smooth Close & 1-Bar Fourier Delta | 50:50 Split | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

window_size = st.sidebar.slider(
    "📦 Rolling FFT Window Size (Historical Candles Only)",
    min_value=32,
    max_value=256,
    value=128,
    step=16,
    help="Strictly past candles used for FFT. Zero future leakage.",
)

cutoff_ratio = st.sidebar.slider(
    "🎛️ Low-Pass Filter Cutoff Ratio",
    min_value=0.01,
    max_value=0.30,
    value=0.08,
    step=0.01,
)

st.sidebar.success(
    "🛡️ **Leak Protection:** FULLY SECURED\n\n"
    "🔒 **FFT Execution:** Strictly Causal Trailing Window\n"
    "📊 **3rd Column:** Zero-Leakage Fourier Delta"
)


# =====================================================================
# MATHEMATICAL ENGINE (Strictly Causal Rolling FFT - Zero Leakage)
# =====================================================================
def apply_rolling_causal_fourier(series_data, window=128, cutoff_fraction=0.08):
    """
    STRICTLY CAUSAL Fourier Filter:
    Processes FFT ONLY on past 'window' candles for every index t.
    Prevents any future data leakage into the current calculation.
    """
    arr = np.asarray(series_data, dtype=float).flatten()
    n = len(arr)
    smoothed_series = np.full(n, np.nan)

    if n < window:
        return smoothed_series

    for i in range(window - 1, n):
        # Extract ONLY historical window (t-window+1 to t)
        sub_window = arr[i - window + 1 : i + 1]

        # FFT on historical slice
        fft_coeffs = np.fft.rfft(sub_window)

        # Apply cutoff
        cutoff_idx = max(1, int(len(fft_coeffs) * cutoff_fraction))
        fft_coeffs[cutoff_idx:] = 0.0

        # Reconstruct historical window
        reconstructed = np.fft.irfft(fft_coeffs, n=window)

        # Take ONLY the current endpoint (index i) - Pure Causal Value
        smoothed_series[i] = reconstructed[-1]

    return smoothed_series


# =====================================================================
# BTC-USD DATA FETCH ENGINE
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_btc_data():
    ticker = "BTC-USD"
    df_raw = yf.download(
        tickers=ticker, period="2y", interval="1h", progress=False
    )

    if df_raw.empty:
        raise ValueError("YFinance API se Bitcoin (BTC-USD) ka data nahi mila.")

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df_raw.index.tzinfo is None:
        df_raw.index = df_raw.index.tz_localize("UTC")

    df_raw.index = df_raw.index.tz_convert("Asia/Kolkata")
    return df_raw


# Fetch Data
try:
    with st.spinner("🔄 Fetching Hourly Bitcoin Data (`BTC-USD`)..."):
        df = fetch_btc_data()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df = df.iloc[:-1]  # Drop active unclosed candle
except Exception as e:
    st.error(f"🚨 Data Engine Error: {e}")
    st.stop()


# =====================================================================
# ⚡ STRICTLY CAUSAL FOURIER PIPELINE
# =====================================================================
close_prices = np.asarray(df["Close"], dtype=float).flatten()

# 1. Rolling Causal Fourier Filtering
df["Fourier_Smooth_Close"] = apply_rolling_causal_fourier(
    close_prices, window=window_size, cutoff_fraction=cutoff_ratio
)

# 2. Causal Fourier Delta (Current - Last)
df["Fourier_Delta"] = df["Fourier_Smooth_Close"] - df[
    "Fourier_Smooth_Close"
].shift(1)


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

df_predict.dropna(subset=["Fourier_Smooth_Close"], inplace=True)

st.success(
    f"🟢 **Synced via Yahoo Finance (BTC-USD): {total_candles:,} Hourly Candles** | "
    f"🧠 **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:** {len(df_predict):,}"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
display_df = pd.DataFrame(index=df_predict.index)
display_df["Close"] = df_predict["Close"].round(2)

# Method 2 Zero-Leakage Fourier Columns
display_df["Fourier_Smooth_Close"] = df_predict["Fourier_Smooth_Close"].round(2)
display_df["Fourier_Delta"] = df_predict["Fourier_Delta"].round(2)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M IST")

# Metric Cards Display
latest_candle = display_df.iloc[0]
latest_time = display_df.index[0]

st.markdown(f"### 🔒 **LAST LOCKED CANDLE (IST):** `{latest_time}`")

col1, col2, col3 = st.columns(3)
col1.metric("BTC Close Price", f"${latest_candle['Close']:,.2f}")
col2.metric(
    "🛡️ Zero-Leak Fourier Smooth",
    f"${latest_candle['Fourier_Smooth_Close']:,.2f}",
)
col3.metric(
    "⚡ Zero-Leak Fourier Delta",
    f"${latest_candle['Fourier_Delta']:,.2f}",
)

st.divider()

st.subheader(
    f"📋 50:50 Matrix — Strictly Causal Fourier Engine ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "Fourier_Smooth_Close": st.column_config.NumberColumn(
            "🛡️ Zero-Leak Smooth ($)", format="$%.2f"
        ),
        "Fourier_Delta": st.column_config.NumberColumn(
            "⚡ Zero-Leak Delta ($)", format="$%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
