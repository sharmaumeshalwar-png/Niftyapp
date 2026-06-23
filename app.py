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
st.set_page_config(page_title="Nifty Column G Matrix (July 2024)", layout="wide")

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
            border-left: 5px solid #f59e0b;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Strict Cascade (July 2024 Available Horizon)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>View Open From:</b> 01 July 2024 (Safe Server Limits)<br>
        <b>Excel Sync Mode:</b> No server data error. Calculations start clean from 1st July 2024 with zero initial drift.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE WITH SAFE HOURLY TIMELINE (JULY 2024)
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_july_2024_synchronized_data():
    try:
        # [SAFE SERVER LIMIT] Setting start to 01 July 2024 for flawless hourly download
        df_raw = yf.download(tickers="^NSEI", start="2024-07-01", interval="1h", progress=False)
        if df_raw.empty:
            return pd.DataFrame()
            
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        
        # Chronological ascending sort (Top row is 01 July 2024)
        df_raw = df_raw.sort_index(ascending=True)
        
        # [EXCEL STABILITY FIX] Dropping the live running hourly row to lock history values
        if len(df_raw) > 1:
            df_raw = df_raw.iloc[:-1].copy()
            
        return df_raw
    except Exception:
        return pd.DataFrame()

df = load_july_2024_synchronized_data()

if not df.empty:
    df = df.reset_index()
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Date_Time'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. ZERO-DRIFT EXCEL MATH VECTOR PIPELINE (ANCHORED AT 01 JULY 2024)
    # ==============================================================================
    # Column A = Mapped strictly to finalized close prices
    df['Column A'] = df['Close'].astype(float)
    
    # Column B Loop - Seed hard-matched on 01 July 2024 row to prevent huge initial gap
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: 
        col_b[0] = float(df['Column A'].values[0]) # Zero-gap initialization anchor
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # Column C = Clean Difference Matrix (Will print 0.0000 on the very first row)
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # Column D = Exponential of C (Seed initialized to C[0])
    col_d = np.zeros(total_rows, dtype=float)
    if total_rows > 0: 
        col_d[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_d[i] = col_d[i-1] + (mul * (float(df['Column C'].values[i]) - col_d[i-1]))
    df['Column D'] = col_d
    
    # Column E = Exponential of D (Seed initialized to D[0])
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: 
        col_e
