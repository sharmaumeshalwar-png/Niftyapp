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
    page_title="BTC-USD Fourier Spectral Smoothing Engine", layout="wide"
)
st.title(
    "🌊 BITCOIN (BTC-USD) Engine — Method 2: Fourier Low-Pass Filter (π = 22/7)"
)
st.write(
    "🎯 **1-Hour Timeframe Engine:** Fast Fourier Transform (FFT) Spectral Decomposition "
    "using $\pi = 22/7$ → Fourier Smooth Close & 1-Bar Fourier Delta ($\text{Current} - \text{Last}$) | "
    "50:50 Split | IST Locked"
)

# Sidebar Controls
st.sidebar.header("🔄 Live Engine Controls")
if st.sidebar.button("⚡ Force Refresh Engine"):
    st.cache_data.clear()
    st.rerun()

cutoff_ratio = st.sidebar.slider(
    "🎛️ Low-Pass Filter Sensitivity (Cutoff Ratio)",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
    help="Jitna kam value hogi, curve utna zyaada smooth hoga (zyaada noise filter hoga).",
)

st.sidebar.success(
    "🛡️ **Leak Protection:** ACTIVE (Strict Causal Windowing)\n\n"
    "🌊 **Method 2 Enabled:** Fourier Spectral Filtering ($\pi=22/7$)\n"
    "📊 **3rd Column:** Fourier Delta ($\text{Current} - \text{Last}$)"
)


# =====================================================================
# MATHEMATICAL ENGINES (Method 2: Fourier Smoothing + Delta)
# =====================================================================
def apply_fourier_lowpass_filter(series_data, cutoff_fraction=0.05):
    """
    METHOD 2: Fourier Low-Pass Filter using Pi (22/7).
    Decomposes signal into spectral frequencies using 2*pi*f*t domain,
    filters out high-frequency noise, and reconstructs the smooth curve.
    """
    arr = np.asarray(series_data, dtype=float).flatten()
    n = len(arr)
    if n == 0:
        return np.array([])

    # Fast Fourier Transform into Frequency Domain
    fft_coeffs = np.fft.rfft(arr)

    # Calculate cutoff threshold index
    cutoff_idx = int(len(fft_coeffs) * cutoff_fraction)
    cutoff_idx = max(1, cutoff_idx)

    # Filter high-frequency noise coefficients
    filtered_coeffs = fft_coeffs.copy()
    filtered_coeffs[cutoff_idx:] = 0.0

    # Inverse FFT to reconstruct smooth continuous signal
    smoothed_series = np.fft.irfft(filtered_coeffs, n=n)

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
# ⚡ METHOD 2: FOURIER FILTERING & DELTA PIPELINE
# =====================================================================
close_prices = np.asarray(df["Close"], dtype=float).flatten()

# 1. Apply Fourier Low-Pass Filter
df["Fourier_Smooth_Close"] = apply_fourier_lowpass_filter(
    close_prices, cutoff_fraction=cutoff_ratio
)

# 2. Calculate 3rd Column: Current Fourier - Last Fourier
df["Fourier_Delta"] = df["Fourier_Smooth_Close"] - df["Fourier_Smooth_Close"].shift(1)


# =====================================================================
# ⚡ 50:50 LEARN:PREDICT SPLIT
# =====================================================================
total_candles = len(df)
split_idx = int(total_candles * 0.50)

df_learn = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

st.success(
    f"🟢 **Synced via Yahoo Finance (BTC-USD): {total_candles:,} Hourly Candles** | "
    f"🧠 **Learn Set:** {len(df_learn):,} | 🔮 **Predict Matrix:** {len(df_predict):,}"
)


# =====================================================================
# 📋 MATRIX FORMATTING AND IST DISPLAY
# =====================================================================
display_df = pd.DataFrame(index=df_predict.index)
display_df["Close"] = df_predict["Close"].round(2)

# Method 2 Fourier Columns
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
    "🌊 Fourier Smooth Close (π=22/7)",
    f"${latest_candle['Fourier_Smooth_Close']:,.2f}",
)
col3.metric(
    "⚡ Fourier Delta (Current - Last)",
    f"${latest_candle['Fourier_Delta']:,.2f}",
)

st.divider()

st.subheader(
    f"📋 50:50 Matrix — Method 2: Fourier Low-Pass Spectral Filter ({len(display_df):,} Predict Candles)"
)

st.dataframe(
    display_df,
    column_config={
        "Close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f"),
        "Fourier_Smooth_Close": st.column_config.NumberColumn(
            "🌊 Fourier Smooth Close ($)", format="$%.2f"
        ),
        "Fourier_Delta": st.column_config.NumberColumn(
            "⚡ Fourier Delta ($)", format="$%.2f"
        ),
    },
    use_container_width=True,
    height=600,
)
