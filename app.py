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
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Pure Cascade (Strict Forward Exponential Engine)</h1>
        <p><b>Interval:</b> 1 Hour (1H) Candles | <b>Data Depth:</b> Frozen 2 Years | <b>View Open From:</b> 01 Jan 2025<br>
        <b>Fixed ValueError:</b> Dynamic columns are explicitly unique. Time is now mapped to <b>Date_Time</b>.</p>
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
    # Time formatting saved explicitly inside an isolated name string
    df['Date_Time'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    mul = 0.0001
    
    # ==============================================================================
    # 3. ADVANCED SEQUENTIAL FORWARD ENGINE (A -> B -> C -> D -> E)
    # ==============================================================================
    # Column A = Strictly Candle Close Price
    df['Column A'] = df['Close'].astype(float)
    
    # Column B = Exponential of A
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (mul * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # Column C = Exponential of B
    col_c = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_c[0] = float(col_b[0])
    for i in range(1, total_rows):
        col_c[i] = col_c[i-1] + (mul * (col_b[i] - col_c[i-1]))
    df['Column C'] = col_c
    
    # Column D = UNIQUE PURE MATHEMATICAL FLOAT VECTOR (Exp of C)
    col_d = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_d[0] = float(col_c[0])
    for i in range(1, total_rows):
        col_d[i] = col_d[i-1] + (mul * (col_c[i] - col_d[i-1]))
    df['Column D'] = col_d
    
    # Column E = D2 - D1 (Velocity rate differential: D(t) - D(t-1))
    col_e = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_e[i] = col_d[i] - col_d[i-1]
    df['Column E'] = col_e
    
    # ==============================================================================
    # 4. DERIVATIVE LAYER MATRIX FOR SIGNALS SYNC
    # ==============================================================================
    # F = Exp of E
    col_f = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_f[0] = float(col_e[0])
    for i in range(1, total_rows):
        col_f[i] = col_f[i-1] + (mul * (col_e[i] - col_f[i-1]))
    df['Column F'] = col_f
    
    # G = Exp of F
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(col_f[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (mul * (col_f[i] - col_g[i-1]))
    df['Column G'] = col_g
    
    # H = Exp of G
    col_h = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_h[0] = float(col_g[0])
    for i in range(1, total_rows):
        col_h[i] = col_h[i-1] + (mul * (col_g[i] - col_h[i-1]))
    df['Column H'] = col_h
    
    # I = Exp of H
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(col_h[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (mul * (col_h[i] - col_i[i-1]))
    df['Column I'] = col_i
    
    # J = Exp of I
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (mul * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    df['Column K'] = np.sign(df['Column F'].values).astype(float) - np.sign(df['Column H'].values).astype(float)
    
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l

    # ==============================================================================
    # 5. COLUMN M MATRIX CONTROLLER
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

    # 6. SIGNALS MATRIX
    sig = ["System Booting"]
    for i in range(1, total_rows):
        k, e_val = float(df['Column K'].values[i]), float(df['Column E'].values[i])
        if e_val > 0:
            sig.append("🟢 SIGN BULLISH" if k in [2.0, 0.0] else "⚠️ 90% CALL TRAP")
        else:
            sig.append("🔴 SIGN BEARISH" if k in [-2.0, 0.0] else "⚠️ 90% PUT TRAP")
    df['Signal_Status'] = sig

    # ==============================================================================
    # 7. SLICE FILTER LOGIC (OPEN DISPLAY FROM 01 JAN 2025)
    # ==============================================================================
    df_f = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if df_f.empty:
        df_f = df.copy() 
        
    if not df_f.empty:
        # Strictly unique columns sequence list matrix layout map
        final_cols = ['Date_Time', 'Column A', 'Column B', 'Column C', 'Column D', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Column M', 'Signal_Status']
        
        show_df = df_f[final_cols].copy().iloc[::-1].reset_index(drop=True)
        flags = df_f['L_Sign_Change'].iloc[::-1].reset_index(drop=True)

        def grid_style(row):
            st_list = [''] * len(row)
            idx = row.name
            
            m_pos = row.index.get_loc('Column M') if 'Column M' in row.index else -1
            s_pos = row.index.get_loc('Signal_Status') if 'Signal_Status' in row.index else -1
            
            if m_pos != -1:
                if idx < len(flags) and flags.iloc[idx]:
                    st_list[m_pos] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1.5px solid #3b82f6;'
                else:
                    st_list[m_pos] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
                    
            if s_pos != -1:
                val = str(row['Signal_Status'])
                if "🟢" in val: st_list[s_pos] = 'background-color: #064e3b; color: #34d399; font-weight: bold;'
                elif "🔴" in val: st_list[s_pos] = 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
                elif "⚠️" in val: st_list[s_pos] = 'background-color: #b45309; color: #fef08a; font-weight: bold;'
                else: st_list[s_pos] = 'background-color: #1f2937; color: #d1d5db;'
                
            return st_list

        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 'Column D': '{:.4f}',
                'Column E': '{:.6f}', 'Column F': '{:.6f}', 'Column G': '{:.6f}',
                'Column H': '{:.6f}', 'Column I': '{:.6f}', 'Column J': '{:.6f}', 
                'Column K': '{:.0f}', 'Column L': '{:.6f}'
            }).apply(grid_style, axis=1),
            use_container_width=True
        )
    else:
        st.warning("No data points verified inside the display window.")
else:
    st.error("Data pipeline load error.")
