import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# 1. BASE MATRIX SETTINGS
# ==============================================================================
st.set_page_config(page_title="Nifty Column M Matrix (Daily)", layout="wide")

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
        <h1>🎯 Nifty 50 Pure Cascade (Daily Candle Theme Controller)</h1>
        <p><b>Interval:</b> Daily Candles (1D) | <b>Start Filter:</b> 01 Jan 2025<br>
        <b>Column M Matrix Rule:</b> If Column L sign changes, Column M turns <b>Absolute Black</b>. Otherwise, stays <b>Pure White</b>.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE (5 YEAR DEPTH FOR PERFECT BACK-CALCULATION)
# ==============================================================================
@st.cache_data(ttl=300)
def load_pure_data():
    # 5y kiya hai taaki 2025 se pehle ka mathematical cascade smooth calculate ho sake
    df_raw = yf.download(tickers="^NSEI", period="5y", interval="1d")
    if df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    time_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. MATHEMATICAL CASCADE ENGINE (8-STEP VERIFICATION COMPLIANT)
    # ==============================================================================
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (mul * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (mul * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (mul * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (mul * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    # 4. COLUMN K & L CORE ENGINE
    df['Column K'] = np.sign(df['Column F'].values).astype(float) - np.sign(df['Column H'].values).astype(float)
    
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # ==============================================================================
    # 5. COLUMN M Engine
    # ==============================================================================
    m_txt = ["➡️ CONTINUOUS"] * total_rows
    chg_flag = np.zeros(total_rows, dtype=bool)
    last_v = 0
    
    for i in range(1, total_rows):
        v = col_l[i]
        curr_s = 1 if v > 0 else (-1 if v < 0 else 0)
        if curr_s != 0:
            if last_v != 0 and curr_s != last_v:
                chg_flag[i] = True
                m_txt[i] = "🔄 SIGN CHANGED"
            last_v = curr_s
            
    df['Column M'] = m_txt
    df['L_Sign_Change'] = chg_flag

    # 6. SIGNALS TRACKER
    sig = ["System Booting"]
    for i in range(1, total_rows):
        k, c = float(df['Column K'].values[i]), float(df['Column C'].values[i])
        if c > 0:
            sig.append("🟢 SIGN BULLISH" if k in [2.0, 0.0] else "⚠️ 90% CALL TRAP")
        else:
            sig.append("🔴 SIGN BEARISH" if k in [-2.0, 0.0] else "⚠️ 90% PUT TRAP")
    df['Signal_Status'] = sig

    # ==============================================================================
    # 7. HIGH-PERFORMANCE RENDERING PIPELINE (UPDATED START FILTER: 2025-01-01)
    # ==============================================================================
    df_f = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_f.empty:
        cols = ['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Column M', 'Signal_Status']
        
        # Chronological Inversion Matrix (Latest on Top)
        show_df = df_f[cols].copy().iloc[::-1].reset_index(drop=True)
        flags = df_f['L_Sign_Change'].iloc[::-1].reset_index(drop=True)

        def grid_style(row):
            idx = row.name
            st_list = [''] * len(row)
            m_pos = row.index.get_loc('Column M')
            s_pos = row.index.get_loc('Signal_Status')
