import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# SYSTEM CALCULATION ENGINE: 8-STEP VERIFICATION MATRIX
# ==============================================================================

# STEP 1: STREAMLIT ENVIRONMENT INITIALIZATION
st.set_page_config(page_title="Advanced Reversal Engine", layout="wide", page_icon="📊")
st.title("📊 Custom Reversal Engine & Market Matrix Dashboard")
st.write("Strict Time Window Lock: **Jan 2025 - Dec 2026** (1-Hour Candles)")

# STEP 2: USER METRIC INPUT PARSING
st.sidebar.header("🎛️ System Parameters (Step 2/8)")
ticker = st.sidebar.text_input("Ticker Name (Yahoo Finance)", value="BTC-USD")
delta_lookback = st.sidebar.number_input("Delta Lookback (Column C)", min_value=1, value=14)
long_roc = st.sidebar.number_input("Coppock Long ROC", min_value=1, value=14)
short_roc = st.sidebar.number_input("Coppock Short ROC", min_value=1, value=11)
wma_smoothing = st.sidebar.number_input("Coppock WMA Smoothing", min_value=1, value=10)

# STEP 3: ANTI-LEAK DATA INGESTION PIPELINE
@st.cache_data
def load_untruncated_data(symbol):
    # Fetching historical raw data block to allow rolling calculation windows safely
    raw_download = yf.download(tickers=symbol, interval="1h", period="max")
    if isinstance(raw_download.columns, pd.MultiIndex):
        raw_download.columns = raw_download.columns.get_level_values(0)
    return raw_download

try:
    raw_data = load_untruncated_data(ticker)
    
    if raw_data.empty:
         st.error("❌ Data download fail ho gaya. Kripya correct Ticker format use karein.")
    else:
        df_calc = raw_data.copy()

        # STEP 4: COLUMN C (DELTA PULSE) MATH LOOP
        df_calc['ColumnC_Delta'] = df_calc['Close'] - df_calc['Close'].shift(int(delta_lookback))

        # STEP 5: COPPOCK HYBRID VECTOR CALCULATIONS
        df_calc['ROC_Long'] = ((df_calc['Close'] - df_calc['Close'].shift(int(long_roc))) / df_calc['Close'].shift(int(long_roc))) * 100
        df_calc['ROC_Short'] = ((df_calc['Close'] - df_calc['Close'].shift(int(short_roc))) / df_calc['Close'].shift(int(short_roc))) * 100
        df_calc['RawMatrixSum'] = df_calc['ROC_Long'] + df_calc['ROC_Short']
        
        # Linear WMA Array Matrix alignment
        weights = np.arange(1, int(wma_smoothing) + 1)
        df_calc['CoppockCurve'] = df_calc['RawMatrixSum'].rolling(int(wma_smoothing)).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

        # STEP 6: EXCEL EXTREME ZONE CLASSIFICATION MATRIX (From Laptop Formula)
        # Mapping your exact laptop conditions: >= 2.5 (Overbought), <= -2.5 (Oversold)
        def classify_excel_zone(val):
            if pd.isna(val):
                return "Normal Zone"
            if val >= 2.5:
                return "⚠️ Extreme Overbought (+2.5)"
            elif val <= -2.5:
                return "🟢 Extreme Oversold (-2.5)"
            else:
                return "Normal Zone"

        df_calc['Excel_Zone'] = df_calc['CoppockCurve'].apply(classify_excel_zone)

        # STEP 7: TEMPORAL INVARIANT STRICT 2-YEAR FREEZE LOCK (Jan 2025 - Dec 2026)
        start_freeze = pd.Timestamp("2025-01-01 00:00:00")
        end_freeze = pd.Timestamp("2026-12-31 23:59:59")
        
        # Isolating rows within coordinates to ensure zero forward/backward leakage
        df = df_calc[(df_calc.index >= start_freeze) & (df_calc.index <= end_freeze)].copy()

        # Signal Crossover Logic applied to filtered bounds
        df['Coppock_Prev'] = df['CoppockCurve'].shift(1)
        df['Bullish_Hint'] = (df['ColumnC_Delta'] > 0) & (df['CoppockCurve'] > 0) & (df['Coppock_Prev'] <= 0)
        df['Bearish_Hint'] = (df['ColumnC_Delta'] < 0) & (df['CoppockCurve'] < 0) & (df['Coppock_Prev'] >= 0)

        # STEP 8: SIGNAL CONVERGENCE & INTERACTIVE UI RENDERING
        if df.empty:
            st.warning("⚠️ Is selected date freeze window (2025-2026) me koi data match nahi hua.")
        else:
            # Candlestick chart processing loop
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price Engine"))
            
            # Plotting Long Points
            longs = df[df['Bullish_Hint']]
            fig.add_trace(go.Scatter(x=longs.index, y=longs['Low'] * 0.995, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=13, color='#00FF00'), name='LONG ENTRY'))
            
            # Plotting Short Points
            shorts = df[df['Bearish_Hint']]
            fig.add_trace(go.Scatter(x=shorts.index, y=shorts['High'] * 1.005, mode='markers', 
                                     marker=dict(symbol='triangle-down', size=13, color='#FF0000'), name='SHORT ENTRY'))
            
            fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # All Possible Outcome Dates - Live Matrix Logs Display
            st.subheader("📋 Verified Logs (All Possible Outcome Dates Matrix)")
            
            # Displaying data with zones and transitions
            display_df = df[['Close', 'ColumnC_Delta', 'CoppockCurve', 'Excel_Zone']].copy()
            
            # Highlight custom conditions
            st.dataframe(display_df.tail(100), use_container_width=True)

except Exception as system_fault:
    st.error(f"⚠️ High-Level Execution Halt at Step 8: {system_fault}")
