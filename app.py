import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(page_title="Custom Reversal Engine (2-Year Freeze)", layout="wide")
st.title("📊 Custom Reversal Engine (Column C + Coppock Matrix)")
st.write("Strict 2-Year Freeze Analysis Window: **Jan 2025 - Dec 2026** | Timeframe: **1 Hour**")

# ==========================================
# SIDEBAR CONTROLS & INPUT PARAMETERS
# ==========================================
st.sidebar.header("🎛️ System Parameters")
ticker = st.sidebar.text_input("Stock/Crypto Ticker (Yahoo Finance)", value="BTC-USD")

delta_lookback = st.sidebar.number_input("Delta Pulse Lookback (Column C)", value=14)
long_roc = st.sidebar.number_input("Coppock Long ROC", value=14)
short_roc = st.sidebar.number_input("Coppock Short ROC", value=11)
wma_smoothing = st.sidebar.number_input("Coppock WMA Smoothing", value=10)

# ==========================================
# DATA FETCHING & 2-YEAR FREEZE FILTER
# ==========================================
@st.cache_data
def load_data(symbol):
    # Fetching historical 1-hour data (Max allowed by yfinance for hourly is 2 years)
    df = yf.download(tickers=symbol, interval="1h", period="max")
    return df

try:
    raw_df = load_data(ticker)
    
    # Reset index to handle multi-level columns if any from yfinance
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    # Strict 2-Year Freeze Window Filter (Jan 2025 to Dec 2026)
    start_date = pd.Timestamp("2025-01-01 00:00:00")
    end_date = pd.Timestamp("2026-12-31 23:59:59")
    
    # Filtering data based on the requested verification dates
    df = raw_df[(raw_df.index >= start_date) & (raw_df.index <= end_date)].copy()

    if df.empty:
        st.error("❌ Is Date Range (Jan 2025 - Dec 2026) ke liye selected ticker ka hourly data available nahi hai. Kripya koi dusra ticker try karein.")
    else:
        # ==========================================
        # ENGINE MATHEMATICAL CALCULATIONS
        # ==========================================
        # 1. Column C (Delta Pulse)
        df['ColumnC_Delta'] = df['Close'] - df['Close'].shift(delta_lookback)

        # 2. Coppock Matrix Calculation
        df['ROC_Long'] = ((df['Close'] - df['Close'].shift(long_roc)) / df['Close'].shift(long_roc)) * 100
        df['ROC_Short'] = ((df['Close'] - df['Close'].shift(short_roc)) / df['Close'].shift(short_roc)) * 100
        df['RawMatrixSum'] = df['ROC_Long'] + df['ROC_Short']
        
        # Linear Weighting for WMA Smoothing
        weights = np.arange(1, wma_smoothing + 1)
        df['CoppockCurve'] = df['RawMatrixSum'].rolling(wma_smoothing).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        # 3. Reversal Signals Logic (Crossovers)
        df['Bullish_Hint'] = (df['ColumnC_Delta'] > 0) & (df['CoppockCurve'] > 0) & (df['CoppockCurve'].shift(1) <= 0)
        df['Bearish_Hint'] = (df['ColumnC_Delta'] < 0) & (df['CoppockCurve'] < 0) & (df['CoppockCurve'].shift(1) >= 0)

        # ==========================================
        # VISUALIZATION (Interactive Plotly Chart)
        # ==========================================
        fig = go.Figure()

        # Candlestick Chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))

        # Add Long Buy Signals
        long_signals = df[df['Bullish_Hint']]
        fig.add_trace(go.Scatter(x=long_signals.index, y=long_signals['Low'] * 0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='LONG HINT'))

        # Add Short Sell Signals
        short_signals = df[df['Bearish_Hint']]
        fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals['High'] * 1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='SHORT HINT'))

        fig.update_layout(title=f"{ticker} 1-Hour Chart Setup", yaxis_title="Price", xaxis_title="Date", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # ALL POSSIBLE OUTCOME DATES MATRIX DISPLAY
        # ==========================================
        st.subheader("📋 Generated Signals Log Matrix (8-Step Verified)")
        signals_df = df[df['Bullish_Hint'] | df['Bearish_Hint']].copy()
        
        if not signals_df.empty:
            signals_df['Signal Type'] = np.where(signals_df['Bullish_Hint'], '🟢 LONG (Reversal)', '🔴 SHORT (Exhaustion)')
            display_cols = ['Close', 'ColumnC_Delta', 'CoppockCurve', 'Signal Type']
            st.dataframe(signals_df[display_cols].tail(50), use_container_width=True)
        else:
            st.info("ℹ️ Is specific parameters lock par abhi tak koi signal generate nahi hua.")

except Exception as e:
    st.error(f"Execution Error: {e}")
