import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ==============================================================================
# 1. BASE MATRIX CONFIGURATION (1-HOUR TIME MATRIX)
# ==============================================================================
st.set_page_config(page_title="Nifty Column M Matrix (1-Hour)", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #111827, #1f2937);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #10b981;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Pure Cascade (Sequential Exponential Engine)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>Data Depth:</b> Frozen 2 Years | <b>View Open From:</b> 01 Jan 2025<br>
        <b>Rule:</b> E is exp of C | F is exp of E | G is exp of F | H is exp of G (Using Multiplier formula same as B).</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE (FLAT HOURLY ENGINE)
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_pure_data_hourly():
    try:
        df_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
        if df_raw.empty:
            return pd.DataFrame()
            
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        return df_raw
    except Exception:
        return pd.DataFrame()

df = load_pure_data_hourly()

if not df.empty:
    df = df.reset_index()
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. SEQUENTIAL EXPONENTIAL CASCADE ENGINE (SAME MULTIPLIER FORMULA)
    # ==============================================================================
    # A = (High + Low) / 2
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    # B = Exp of A
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # C = A - B
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # E = Exp of C
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (mul * (float(df['Column C'].values[i]) - col_e[i-1]))
    df
