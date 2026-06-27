import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50: Advanced Matrix Terminal")
st.write("Layout: A=Close, B=TRIX, C=Residue (Minus Allowed), D=RAVI, E=Action Signal")

# ==========================================
# STEP 1: SAFE DATA INGESTION
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
        return pd.DataFrame({"Close": np.linspace(23000, 24000, 100)}, index=dates)

raw_data = load_nifty_data()

if raw_data.empty:
    st.error("Data load nahi ho paaye. Please refresh karein.")
else:
    # ==========================================
    # STEP 2 & 3: COLUMN A & B (TRIX)
    # ==========================================
    raw_data['A'] = raw_data['Close']
    
    trix_period = 15
    log_a = np.log(raw_data['A'])
    ema1 = log_a.ewm(span=trix_period, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_period, adjust=False).mean()
    
    raw_data['B'] = np.exp(ema3)
    
    # ==========================================
    # STEP 4 & 5: COLUMN C (Minus Allowed) & D
    # ==========================================
    # C wapas minus me allow kar diya hai
    raw_data['C'] = raw_data['A'] - raw_data['B']
    
    short_span = 7
    long_span = 65
    
    ema_short_c = raw_data['C'].ewm(span=short_span, adjust=False).mean()
    ema_long_c = raw_data['C'].ewm(span=long_span, adjust=False).mean()
    
    raw_data['D'] = ema_short_c - ema_long_c

    # ==========================================
    # STEP 6: NEW COLUMN E (Trading Signal Engine)
    # ==========================================
    raw_data['E'] = "HOLD"
    raw_data.loc[(raw_data['C'] > 0) & (raw_data['D'] > 0), 'E'] = "🟢 BUY (Bullish)"
    raw_data.loc[(raw_data['C'] < 0) & (raw_data['D'] < 0), 'E'] = "🔴 SELL (Bearish)"

    raw_data.dropna(subset=['B', 'C', 'D'], inplace=True)

    # ==========================================
    # STEP 7: FORMATTING & REVERSE SORTING
    # ==========================================
    output_matrix = raw_data[['A', 'B', 'C', 'D', 'E']].copy()
    
    output_matrix['A'] = output_matrix['A'].map(lambda x: f"{x:.2f}")
    output_matrix['B'] = output_matrix['B'].map(lambda x: f"{x:.2f}")
    output_matrix['C'] = output_matrix['C'].map(lambda x: f"{x:.4f}") # Sign automatic show hoga
    output_matrix['D'] = output_matrix['D'].map(lambda x: f"{x:.4f}")
    
    output_matrix = output_matrix.sort_index(ascending=False)

    # Main Grid output display
    st.subheader("📋 Core Mathematical Matrix")
    st.dataframe(output_matrix, use_container_width=True)

    st.success("Reset Complete: Column C ab true negative values dikhayega jab price TRIX ke neeche hoga.")
