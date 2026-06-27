import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50: TRIX Matrix Engine")
st.write("Layout: A=Close, B=TRIX Price Proxy, C=Residue (A-B), D=Markov State (1-Hour Candles)")

# ==========================================
# STEP 1: SAFE DATA INGESTION (1-Hour Candles)
# ==========================================
@st.cache_data
def load_nifty_data():
    try:
        ticker = "^NSEI"
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except:
        dates = pd.date_range(start="2025-01-01", periods=100, freq="h")
        return pd.DataFrame({"Close": np.linspace(23000, 24000, 100), "High": 24050, "Low": 22950}, index=dates)

raw_data = load_nifty_data()

if raw_data.empty:
    st.error("Data load nahi ho paaye. Please refresh karein.")
else:
    # ==========================================
    # STEP 2 & 3: TRIX AND MATRIX B COMPUTATION
    # ==========================================
    # A = Close Price
    raw_data['A'] = raw_data['Close']
    
    # TRIX Mathematical Calculation: Triple Smoothing
    period = 15
    log_a = np.log(raw_data['A'])
    ema1 = log_a.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # TRIX standard formula percent change hota hai, hum use price scale standard proxy banayenge
    # B = Exponential value of Triple EMA scaled back to price
    raw_data['B'] = np.exp(ema3)
    
    # C = A - B (Residue Wave)
    raw_data['C'] = raw_data['A'] - raw_data['B']
    
    # ==========================================
    # STEP 4 & 5: FORMULA D (TRIX Based States)
    # ==========================================
    rolling_std = raw_data['C'].rolling(window=20).std()
    mean_c = raw_data['C'].mean()
    
    # State Matrix D mapping based on TRIX residue deviation
    raw_data['D'] = "State 2 (Trend Equilibrium)"
    raw_data.loc[raw_data['C'] > (mean_c + 0.8 * rolling_std), 'D'] = "State 1 (TRIX Momentum Bullish)"
    raw_data.loc[raw_data['C'] < (mean_c - 0.8 * rolling_std), 'D'] = "State 0 (TRIX Momentum Bearish)"
    
    raw_data.dropna(subset=['B', 'C', 'D'], inplace=True)

    # ==========================================
    # STEP 6 & 7: FORMATTING & SORTING (Latest First)
    # ==========================================
    output_matrix = raw_data[['A', 'B', 'C', 'D']].copy()
    
    output_matrix['A'] = output_matrix['A'].map(lambda x: f"{x:.2f}")
    output_matrix['B'] = output_matrix['B'].map(lambda x: f"{x:.2f}")
    output_matrix['C'] = output_matrix['C'].map(lambda x: f"{x:.4f}")
    
    # Reverse sorting to get latest candle on top
    output_matrix = output_matrix.sort_index(ascending=False)

    # Display Table Grid
    st.subheader("📋 TRIX Mathematical Log Matrix")
    st.dataframe(output_matrix, use_container_width=True)

    # ==========================================
    # STEP 8: STATISTICAL ANALYSIS SUMMARY TABLE
    # ==========================================
    st.subheader("📊 TRIX State Aggregates")
    summary_data = []
    states_list = ["State 0 (TRIX Momentum Bearish)", "State 1 (TRIX Momentum Bullish)", "State 2 (Trend Equilibrium)"]
    
    for state in states_list:
        subset = raw_data[raw_data['D'] == state]
        summary_data.append({
            "TRIX State (D)": state,
            "Total Hours Observed": len(subset),
            "Average TRIX Residue (C)": f"{subset['C'].mean():.4f}" if len(subset) > 0 else "0.0000"
        })
    st.table(pd.DataFrame(summary_data))
