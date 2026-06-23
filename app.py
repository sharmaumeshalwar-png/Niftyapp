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
st.set_page_config(page_title="Nifty Column G Matrix (Stable)", layout="wide")

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
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Strict Cascade (Excel-Stabilized Freeze Engine)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>Data Depth:</b> Frozen 2 Years | <b>View Open From:</b> 01 Jan 2026<br>
        <b>Stability Mode:</b> Completed candles are frozen. Current incomplete hour candle is dropped to mirror Excel matching.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE (EXCEL-STABILIZED ENGINE)
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_pure_data_hourly_stable():
    try:
        df_raw = yf.download(tickers="^NSEI", period="2y", interval="1h", progress=False)
        if df_raw.empty:
            return pd.DataFrame()
            
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        
        # [STABILITY FIX] Chronological sorting step 
        df_raw = df_raw.sort_index(ascending=True)
        
        # [EXCEL MATCHING BLOCK] Dropping the last running incomplete candle row to freeze calculations
        if len(df_raw) > 1:
            df_raw = df_raw.iloc[:-1].copy()
            
        return df_raw
    except Exception:
        return pd.DataFrame()

df = load_pure_data_hourly_stable()

if not df.empty:
    df = df.reset_index()
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Date_Time'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. BALANCED VECTOR EXCEL SEQUENCE (A -> B -> C -> D -> E -> F -> G)
    # ==============================================================================
    # Column A = Mapped strictly to finalized close prices
    df['Column A'] = df['Close'].astype(float)
    
    # Column B = Exponential of A (Standard Excel Anchor System)
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # Column C = Constant Differential Row Array
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # Column D = Exponential of C 
    col_d = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_d[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_d[i] = col_d[i-1] + (mul * (float(df['Column C'].values[i]) - col_d[i-1]))
    df['Column D'] = col_d
    
    # Column E = Exponential of D
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(col_d[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (mul * (col_d[i] - col_e[i-1]))
    df['Column E'] = col_e
    
    # Column F = Exponential of E
    col_f = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_f[0] = float(col_e[0])
    for i in range(1, total_rows):
        col_f[i] = col_f[i-1] + (mul * (col_e[i] - col_f[i-1]))
    df['Column F'] = col_f
    
    # Column G = F(t) - F(t-1) (Excel Row Line Velocity Tracker)
    col_g = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_g[i] = col_f[i] - col_f[i-1]
    df['Column G'] = col_g

    # ==============================================================================
    # 4. DISPLAY FRAMEWORK FILTER BLOCK (STRICTLY OPEN DISPLAY FROM 01 JAN 2026)
    # ==============================================================================
    df_f = df[df['Raw_Date'] >= '2026-01-01'].copy()
    
    if df_f.empty:
        df_f = df.copy() 
        
    if not df_f.empty:
        # Constraining output sequence matching up to Column G fields
        final_cols = ['Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G']
        
        # Inverting dataframe order dynamically for modern grid UI visibility (Latest on top)
        show_df = df_f[final_cols].copy().iloc[::-1].reset_index(drop=True)

        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
                'Column D': '{:.4f}', 'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.6f}'
            }),
            use_container_width=True
        )
    else:
        st.warning("No completed historical rows found inside the specified 2026 frame.")
else:
    st.error("Data pipeline load error.")
